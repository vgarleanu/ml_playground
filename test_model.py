#!/usr/bin/env python3
import os
import torch
import json
from typing import Dict, Any
from crf_lstm_model import (
    BiLSTM_CRF, extract_metadata, DEVICE, METADATA_FIELDS
)
from cli import load_crf_model

def test_model():
    """Run a comprehensive test of the model against a variety of filename patterns"""
    # Load the trained model
    print("Loading model...")
    model_path = "crf_model_output/model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model, char_to_idx, idx_to_tag = load_crf_model(model_path)

    # Test cases from torrent-name-parser
    test_cases = [
        {
            "filename": "[ www.UsaBit.com ] - My Cousin Vinny (1992) BluRay 720p 750MB Ganool",
            "expected": {"title": "My Cousin Vinny", "year": "1992"}
        },
        {
            "filename": "2012.2009.1080p.BluRay.x264.DTS-METiS",
            "expected": {"title": "2012", "year": "2009"}
        },
        {
            "filename": "[TorrentCounter.to].Pacific.Rim.2.Uprising.2018.1080p.HC.HDRip.x264.[2GB]",
            "expected": {"title": "Pacific Rim 2 Uprising", "year": "2018"}
        },
        {
            "filename": "Blade.Runner.2049.2017.HDRip",
            "expected": {"title": "Blade Runner 2049", "year": "2017"}
        },
        {
            "filename": "Euphoria.US.S01E03.Made.You.Look.1080p.AMZN.WEB-DL.DDP5.1.H.264-KiNGS",
            "expected": {"title": "Euphoria", "season": "1", "episode": "3"}
        },
        {
            "filename": "narcos.s01e10.1080p.bluray.x264-rovers",
            "expected": {"title": "narcos", "season": "1", "episode": "10"}
        },
        {
            "filename": "Rome.S01E11.The.Spoils.BluRay.10Bit.1080p.Dts.H265-d3g",
            "expected": {"title": "Rome", "season": "1", "episode": "11"}
        },
        {
            "filename": "the.expanse.s01e09e10.1080p.bluray.x264-rovers",
            "expected": {"title": "the expanse", "season": "1", "episode": "9"}
        },
        {
            "filename": "Attack on Titan (Shingeki no Kyojin) Season 2 [1080p x265 10bit BD Dual Audio AAC]/Episode 30 - Historia",
            "expected": {"title": "Attack on Titan", "season": "2", "episode": "30"}
        },
        {
            "filename": "The Walking Dead S05E03 720p HDTV x264-ASAP[ettv]",
            "expected": {"title": "The Walking Dead", "season": "5", "episode": "3"}
        },
        {
            "filename": "Hercules (2014) 1080p BrRip H264 - YIFY",
            "expected": {"title": "Hercules", "year": "2014"}
        },
        {
            "filename": "Dawn.of.the.Planet.of.the.Apes.2014.HDRip.XViD-EVO",
            "expected": {"title": "Dawn of the Planet of the Apes", "year": "2014"}
        },
        {
            "filename": "The Big Bang Theory S08E06 HDTV XviD-LOL [eztv]",
            "expected": {"title": "The Big Bang Theory", "season": "8", "episode": "6"}
        },
        {
            "filename": "22 Jump Street (2014) 720p BrRip x264 - YIFY",
            "expected": {"title": "22 Jump Street", "year": "2014"}
        },
        {
            "filename": "Hercules.2014.EXTENDED.1080p.WEB-DL.DD5.1.H264-RARBG",
            "expected": {"title": "Hercules", "year": "2014"}
        },
        {
            "filename": "Hercules.2014.EXTENDED.HDRip.XViD-juggs[ETRG]",
            "expected": {"title": "Hercules", "year": "2014"}
        },
        {
            "filename": "Hercules (2014) WEBDL DVDRip XviD-MAX",
            "expected": {"title": "Hercules", "year": "2014"}
        },
        {
            "filename": "WWE Hell in a Cell 2014 PPV WEB-DL x264-WD -={SPARROW}=-",
            "expected": {"title": "WWE Hell in a Cell", "year": "2014"}
        },
        {
            "filename": "UFC.179.PPV.HDTV.x264-Ebi[rartv]",
            "expected": {"title": "UFC 179"}
        },
        {
            "filename": "Marvels Agents of S H I E L D S02E05 HDTV x264-KILLERS [eztv]",
            "expected": {"title": "Marvels Agents of S H I E L D", "season": "2", "episode": "5"}
        },
        {
            "filename": "X-Men.Days.of.Future.Past.2014.1080p.WEB-DL.DD5.1.H264-RARBG",
            "expected": {"title": "X-Men Days of Future Past", "year": "2014"}
        },
        {
            "filename": "Guardians Of The Galaxy 2014 R6 720p HDCAM x264-JYK",
            "expected": {"title": "Guardians Of The Galaxy", "year": "2014"}
        },
        {
            "filename": "Marvel\\'s.Agents.of.S.H.I.E.L.D.S02E01.Shadows.1080p.WEB-DL.DD5.1",
            "expected": {"title": "Marvel's Agents of S H I E L D", "season": "2", "episode": "1"}
        },
        {
            "filename": "Marvels Agents of S.H.I.E.L.D. S02E06 HDTV x264-KILLERS[ettv]",
            "expected": {"title": "Marvels Agents of S H I E L D", "season": "2", "episode": "6"}
        },
        {
            "filename": "Guardians of the Galaxy (CamRip / 2014)",
            "expected": {"title": "Guardians of the Galaxy", "year": "2014"}
        },
        {
            "filename": "The.Walking.Dead.S05E03.1080p.WEB-DL.DD5.1.H.264-Cyphanix[rartv]",
            "expected": {"title": "The Walking Dead", "season": "5", "episode": "3"}
        },
        {
            "filename": "Brave.2012.R5.DVDRip.XViD.LiNE-UNiQUE",
            "expected": {"title": "Brave", "year": "2012"}
        },
        {
            "filename": "Lets.Be.Cops.2014.BRRip.XViD-juggs[ETRG]",
            "expected": {"title": "Lets Be Cops", "year": "2014"}
        },
        {
            "filename": "These.Final.Hours.2013.WBBRip XViD",
            "expected": {"title": "These Final Hours", "year": "2013"}
        },
        {
            "filename": "Downton Abbey 5x06 HDTV x264-FoV [eztv]",
            "expected": {"title": "Downton Abbey", "season": "5", "episode": "6"}
        },
        {
            "filename": "Annabelle.2014.HC.HDRip.XViD.AC3-juggs[ETRG]",
            "expected": {"title": "Annabelle", "year": "2014"}
        },
        {
            "filename": "Lucy.2014.HC.HDRip.XViD-juggs[ETRG]",
            "expected": {"title": "Lucy", "year": "2014"}
        },
        {
            "filename": "The Flash 2014 S01E04 HDTV x264-FUM[ettv]",
            "expected": {"title": "The Flash", "year": "2014", "season": "1", "episode": "4"}
        },
        {
            "filename": "South Park S18E05 HDTV x264-KILLERS [eztv]",
            "expected": {"title": "South Park", "season": "18", "episode": "5"}
        },
        {
            "filename": "The Flash 2014 S01E03 HDTV x264-LOL[ettv]",
            "expected": {"title": "The Flash", "year": "2014", "season": "1", "episode": "3"}
        },
        {
            "filename": "The Flash 2014 S01E01 HDTV x264-LOL[ettv]",
            "expected": {"title": "The Flash", "year": "2014", "season": "1", "episode": "1"}
        },
        {
            "filename": "Lucy 2014 Dual-Audio WEBRip 1400Mb",
            "expected": {"title": "Lucy", "year": "2014"}
        },
        {
            "filename": "Teenage Mutant Ninja Turtles (HdRip / 2014)",
            "expected": {"title": "Teenage Mutant Ninja Turtles", "year": "2014"}
        },
        {
            "filename": "Teenage Mutant Ninja Turtles (unknown_release_type / 2014)",
            "expected": {"title": "Teenage Mutant Ninja Turtles", "year": "2014"}
        },
        {
            "filename": "The Simpsons S26E05 HDTV x264 PROPER-LOL [eztv]",
            "expected": {"title": "The Simpsons", "season": "26", "episode": "5"}
        },
        {
            "filename": "To.All.The.Boys.Always.And.Forever.2021.1080p.NF.WEB-DL.x265.10bit.HDR.DDP5.1.Atmos-NWD",
            "expected": {"title": "To All The Boys Always And Forever", "year": "2021"}
        },
        {
            "filename": "The EXPANSE - S03 E01 - Fight or Flight (1080p - AMZN Web-DL)",
            "expected": {"title": "The EXPANSE", "season": "3", "episode": "1"}
        },
        {
            "filename": "[Judas] Re Zero 2020 - S01E01",
            "expected": {"title": "Re Zero", "year": "2020", "season": "1", "episode": "1"}
        },
        {
            "filename": "Fargo.S04E03.WEB.x264-PHOENiX[TGx]",
            "expected": {"title": "Fargo", "season": "4", "episode": "3"}
        },
        {
            "filename": "[SubsPlease] Dr. Stone S2 - 07 (1080p) [33538C7C]",
            "expected": {"title": "Dr Stone", "season": "2", "episode": "7"}
        },
        {
            "filename": "[SubsPlease] Fumetsu no Anata e S2 - 01 (1080p) [1D65E30D]",
            "expected": {"title": "Fumetsu no Anata e", "season": "2", "episode": "1"}
        },
        {
            "filename": "A Shaun the Sheep Movie - Farmageddon (2019) [h265 Remux-1080p] [tt6193408]",
            "expected": {"title": "A Shaun the Sheep Movie Farmageddon", "year": "2019"}
        },
        {
            "filename": "Yes Day (2021) [h265 WEBDL-1080p] [tt8521876]",
            "expected": {"title": "Yes Day", "year": "2021"}
        }
    ]

    # Run tests and collect results
    results = {
        "passed": 0,
        "failed": 0,
        "total": len(test_cases),
        "details": []
    }

    print(f"Testing model on {len(test_cases)} test cases...")
    for i, test_case in enumerate(test_cases):
        filename = test_case["filename"]
        expected = test_case["expected"]
        
        # Extract metadata using the model
        metadata, _ = extract_metadata(model, char_to_idx, idx_to_tag, filename)
        
        # Check if extraction matches expected values
        test_passed = True
        failures = []
        
        for field, expected_value in expected.items():
            if field not in metadata:
                test_passed = False
                failures.append(f"{field}: missing (expected '{expected_value}')")
            # For numeric fields like season/episode, compare the numeric value not string format
            elif field in ['season', 'episode'] and metadata[field] and expected_value:
                # Convert both to integers for comparison to handle zero-padding differences
                try:
                    extracted_num = int(metadata[field])
                    expected_num = int(expected_value)
                    if extracted_num != expected_num:
                        test_passed = False
                        failures.append(f"{field}: got '{metadata[field]}', expected '{expected_value}'")
                except (ValueError, TypeError):
                    # If conversion fails, fall back to string comparison
                    if str(metadata[field]) != str(expected_value):
                        test_passed = False
                        failures.append(f"{field}: got '{metadata[field]}', expected '{expected_value}'")
            elif str(metadata[field]) != str(expected_value):
                test_passed = False
                failures.append(f"{field}: got '{metadata[field]}', expected '{expected_value}'")
        
        # Record result
        test_result = {
            "filename": filename,
            "passed": test_passed,
            "extracted": metadata,
            "expected": expected,
            "failures": failures
        }
        results["details"].append(test_result)
        
        if test_passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Print progress
        print(f"[{i+1}/{len(test_cases)}] {'✅' if test_passed else '❌'} {filename}")

    # Print summary
    accuracy = results["passed"] / results["total"] * 100
    print("\n" + "="*40)
    print(f"Test completed: {results['passed']}/{results['total']} passed ({accuracy:.2f}% accuracy)")
    print("="*40)
    
    # Show failures
    print("\nFailed test cases:")
    for i, result in enumerate(results["details"]):
        if not result["passed"]:
            print(f"\n{i+1}. {result['filename']}")
            print(f"   Extracted: {result['extracted']}")
            print(f"   Expected:  {result['expected']}")
            print(f"   Failures:  {', '.join(result['failures'])}")
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to test_results.json")

if __name__ == "__main__":
    test_model()