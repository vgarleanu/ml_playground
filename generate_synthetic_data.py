#!/usr/bin/env python3
import random
import csv
import re
from datetime import datetime, timedelta

# Sample data for filling templates
titles = [
    # Standard titles
    "The Shawshank Redemption", "The Godfather", "Pulp Fiction", "The Dark Knight", 
    "Fight Club", "Inception", "Interstellar", "Parasite", "Avengers Endgame", 
    "The Matrix", "Breaking Bad", "Game of Thrones", "Stranger Things", "The Office", 
    "Friends", "The Mandalorian", "Attack on Titan", "Demon Slayer", "My Hero Academia",
    "Jujutsu Kaisen", "One Piece", "Naruto", "Death Note", "Fullmetal Alchemist",
    
    # Titles with numbers - edge cases
    "2001 A Space Odyssey", "21 Jump Street", "22 Jump Street", "300", "1917", "2012", 
    "District 9", "The Magnificent Seven", "Ocean's Eleven", "10 Cloverfield Lane",
    "127 Hours", "500 Days of Summer", "8 Mile", "28 Days Later", "9",
    
    # Titles with future dates
    "Blade Runner 2049", "Death Race 2000", "Space 1999", "Marvel 2099",
    
    # Titles with dots/periods
    "S.W.A.T", "Agents of S.H.I.E.L.D", "M.A.S.H", "G.I. Joe", "E.T."
]

anime_studios = ["Erai-raws", "HorribleSubs", "SubsPlease", "Judas", "MTBB", "Commie", "Underwater"]
release_groups = ["SPARKS", "RARBG", "FraMeSToR", "GROUP", "YIFY", "VEXT", "AMIABLE", "GECKOS"]
years = list(range(1980, 2025))
resolutions = ["720p", "1080p", "2160p", "1440p", "480p"]
sources = ["BluRay", "WEB-DL", "WEBRip", "HDTV", "DVDRip", "BDRip", "AMZN", "NETFLIX", "HULU"]
codecs = ["x264", "x265", "HEVC", "H.264", "AVC", "XVID"]
audio = ["DD5.1", "AAC", "DTS", "TrueHD.7.1", "FLAC", "MP3", "DDP5.1", "AC3", "DTS-HD.MA.5.1"]
extras = ["PROPER", "REPACK", "EXTENDED", "DC", "UNRATED", "IMAX", "REMUX", "10bit", "HDR", "REMASTERED"]
languages = ["KOREAN", "JAPANESE", "FRENCH", "GERMAN", "SPANISH", "ITALIAN", "SWEDISH", "RUSSIAN"]
delimiters = [".", "_", "-", " "]
episode_titles = ["Pilot", "The Beginning", "New Dawn", "Revelations", "The End", "Aftermath", "Origins", "Legacy"]

def clean_title(title, delimiter):
    """Convert a title to torrent filename format with the given delimiter"""
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    return delimiter.join(title.split())

def generate_movie_filename():
    """Generate a synthetic movie filename"""
    title = random.choice(titles)
    year = random.choice(years)
    delimiter = random.choice(delimiters)
    resolution = random.choice(resolutions)
    source = random.choice(sources)
    codec = random.choice(codecs)
    group = random.choice(release_groups)
    
    # Clean the title
    clean_t = clean_title(title, delimiter)
    
    # Basic components
    components = [clean_t, str(year), resolution, source, codec]
    
    # Random extras
    if random.random() < 0.3:
        components.insert(2, random.choice(extras))
    
    # Random audio info
    if random.random() < 0.4:
        components.insert(4, random.choice(audio))
    
    # Random language
    if random.random() < 0.2:
        components.insert(2, random.choice(languages))
    
    # Add release group with varying formats
    if random.random() < 0.7:
        components.append(f"{random.choice(['-', '.'])}{group}")
    
    # Join with delimiter
    filename = delimiter.join(components) + ".mkv"
    
    # Clean up any double delimiters
    filename = re.sub(f"{delimiter}{delimiter}+", delimiter, filename)
    
    # Label components for training
    labels = {
        "media_type": "movie",
        "title": title,
        "year": year,
        "resolution": resolution,
        "source": source if source in components else "",
        "codec": codec if codec in components else "",
        "language": next((lang for lang in languages if lang in components), ""),
    }
    
    return filename, labels

def generate_tv_filename():
    """Generate a synthetic TV show filename with diverse season/episode patterns"""
    title = random.choice(titles)
    season = random.randint(1, 35)  # Wider range of seasons
    episode = random.randint(1, 99)  # Wider range of episodes
    year = random.choice(years)
    delimiter = random.choice(delimiters)
    resolution = random.choice(resolutions)
    source = random.choice(sources)
    codec = random.choice(codecs)
    group = random.choice(release_groups)
    
    # Clean the title
    clean_t = clean_title(title, delimiter)
    
    # Choose episode format with much more diversity
    ep_format_type = random.choices([
        "standard",    # S01E02
        "separator",   # 1x02, 1.02
        "text",        # Season 1 Episode 2
        "dash",        # - 01
        "bracket",     # [01]
        "date",        # 2023.01.01
        "episode_only" # Episode 01, just E01
    ], weights=[40, 15, 15, 10, 5, 10, 5], k=1)[0]
    
    # Generate the episode format based on the selected type
    if ep_format_type == "standard":
        # Standard format: S01E02, s1e2, etc.
        s_prefix = random.choice(["S", "s"])
        e_prefix = random.choice(["E", "e"])
        zero_pad_season = random.choice([True, False])  # Whether to use S01 or S1
        zero_pad_episode = random.choice([True, True, False])  # Bias toward zero padding
        
        if zero_pad_season:
            s_part = f"{s_prefix}{season:02d}"
        else:
            s_part = f"{s_prefix}{season}"
            
        if zero_pad_episode:
            e_part = f"{e_prefix}{episode:02d}"
        else:
            e_part = f"{e_prefix}{episode}"
            
        ep_format = f"{s_part}{e_part}"
        
        # Multi-episode format (10% chance)
        if random.random() < 0.1:
            next_ep = episode + 1
            if zero_pad_episode:
                ep_format += f"{random.choice(['-', e_prefix])}{next_ep:02d}"
            else:
                ep_format += f"{random.choice(['-', e_prefix])}{next_ep}"
    
    elif ep_format_type == "separator":
        # Separator format: 1x02, 1.02, etc.
        sep = random.choice(["x", ".", "-"])
        zero_pad_season = random.choice([False, False, True])  # Bias against zero padding for season
        zero_pad_episode = random.choice([True, True, False])  # Bias toward zero padding for episode
        
        if zero_pad_season:
            s_part = f"{season:02d}"
        else:
            s_part = f"{season}"
            
        if zero_pad_episode:
            e_part = f"{episode:02d}"
        else:
            e_part = f"{episode}"
            
        ep_format = f"{s_part}{sep}{e_part}"
    
    elif ep_format_type == "text":
        # Text format: Season 1 Episode 2, etc.
        s_text = random.choice(["Season", "Season", "SEASON", "Series"])
        e_text = random.choice(["Episode", "Episode", "EPISODE", "Ep", "EP"])
        
        # Format variations
        if random.choice([True, False]):
            # With space: Season 1 Episode 2
            ep_format = f"{s_text} {season} {e_text} {episode}"
        else:
            # Without space: Season1Episode2
            ep_format = f"{s_text}{season}{e_text}{episode}"
    
    elif ep_format_type == "dash":
        # Dash format used in anime: " - 01"
        zero_pad = random.choice([True, True, False])  # Bias toward zero padding
        if zero_pad:
            ep_format = f" - {episode:02d}"
        else:
            ep_format = f" - {episode}"
    
    elif ep_format_type == "bracket":
        # Bracket format: [01], [E01], etc.
        zero_pad = random.choice([True, True, False])
        use_e_prefix = random.choice([False, True])
        
        if zero_pad:
            if use_e_prefix:
                ep_format = f"[E{episode:02d}]"
            else:
                ep_format = f"[{episode:02d}]"
        else:
            if use_e_prefix:
                ep_format = f"[E{episode}]"
            else:
                ep_format = f"[{episode}]"
    
    elif ep_format_type == "date":
        # Date format: 2023.01.01
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date_sep = random.choice([".", "-", "_"])
        
        ep_format = f"{year}{date_sep}{month:02d}{date_sep}{day:02d}"
    
    elif ep_format_type == "episode_only":
        # Episode only format: Episode 01, E01
        if random.choice([True, False]):
            # Text format
            ep_format = f"Episode {episode:02d}"
        else:
            # E-prefix only
            ep_format = f"E{episode:02d}"
    
    # Episode title (50% chance)
    ep_title = ""
    if random.random() < 0.5:
        ep_title = clean_title(random.choice(episode_titles), delimiter)
    
    # Basic components
    components = [clean_t, ep_format]
    
    # Add episode title if present
    if ep_title:
        components.append(ep_title)
    
    # Add remaining components
    components.extend([resolution, source, codec])
    
    # Random extras
    if random.random() < 0.3:
        components.insert(3, random.choice(extras))
    
    # Random audio info
    if random.random() < 0.4:
        components.insert(4, random.choice(audio))
    
    # Add release group with varying formats
    if random.random() < 0.7:
        components.append(f"{random.choice(['-', '.'])}{group}")
    
    # Join with delimiter
    filename = delimiter.join(components) + ".mkv"
    
    # Clean up any double delimiters
    filename = re.sub(f"{delimiter}{delimiter}+", delimiter, filename)
    
    # Extract the actual number values for training
    # We store just the numeric part without S/E prefixes
    if ep_format_type in ["standard", "separator", "text", "episode_only"]:
        season_value = season
        episode_value = episode
    elif ep_format_type == "dash" or ep_format_type == "bracket":
        season_value = ""  # Often no season marker in these formats
        episode_value = episode
    elif ep_format_type == "date":
        season_value = ""
        episode_value = ""  # No clear episode number for date format
    
    # Label components for training
    labels = {
        "media_type": "tv",
        "title": title,
        "season": season_value if season_value != "" else "",
        "episode": episode_value if episode_value != "" else "",
        "episode_title": random.choice(episode_titles) if ep_title else "",
        "resolution": resolution,
        "source": source if source in components else "",
    }
    
    return filename, labels

def generate_anime_filename():
    """Generate a synthetic anime filename with diverse patterns"""
    title = random.choice(titles)
    episode = random.randint(1, 99)  # Wider range of episodes
    
    # More diverse season representation
    season_format = random.choices([
        "none",     # No season info
        "number",   # S2, Season 2
        "ordinal",  # Second Season
        "sequel",   # Title II, Title 2nd Season
    ], weights=[50, 30, 10, 10], k=1)[0]
    
    if season_format == "none":
        season = ""
        season_value = ""
    elif season_format == "number":
        season_num = random.randint(2, 5)  # Season numbers 2-5
        season_prefix = random.choice(["S", "Season "])
        season = f"{season_prefix}{season_num}"
        season_value = season_num
    elif season_format == "ordinal":
        season_num = random.randint(2, 5)
        ordinals = ["Second", "Third", "Fourth", "Fifth"]
        season = f"{ordinals[season_num-2]} Season"
        season_value = season_num
    else:  # sequel
        season_num = random.randint(2, 4)
        sequel_format = random.choice([
            f"{title} II", 
            f"{title} {season_num}nd Season",
            f"{title} Part {season_num}"
        ])
        # For sequel formats, the season is part of the title
        title = sequel_format
        season = ""
        season_value = season_num
    
    resolution = random.choice(resolutions)
    studio = random.choice(anime_studios)
    
    # More diverse episode number formats
    ep_format_type = random.choices([
        "standard",    # - 01
        "bracket",     # [01]
        "abbrev",      # EP01, Ep.01
        "text",        # Episode 01
        "hashtag",     # #01
    ], weights=[60, 15, 10, 10, 5], k=1)[0]
    
    if ep_format_type == "standard":
        # Standard anime format: " - 01"
        zero_pad = random.choice([True, True, False])  # Bias toward zero padding
        if zero_pad:
            ep_part = f" - {episode:02d}"
        else:
            ep_part = f" - {episode}"
    
    elif ep_format_type == "bracket":
        # Bracket format: [01], [EP01]
        zero_pad = random.choice([True, True, False])
        use_prefix = random.choice([False, True])
        
        if zero_pad:
            if use_prefix:
                ep_part = f" [EP{episode:02d}]"
            else:
                ep_part = f" [{episode:02d}]"
        else:
            if use_prefix:
                ep_part = f" [EP{episode}]"
            else:
                ep_part = f" [{episode}]"
    
    elif ep_format_type == "abbrev":
        # Abbreviated: EP01, Ep.01
        prefix = random.choice(["EP", "Ep.", "ep", "Episode"])
        ep_part = f" {prefix}{episode:02d}"
        
    elif ep_format_type == "text":
        # Text format: Episode 01
        ep_part = f" Episode {episode:02d}"
        
    else:  # hashtag
        # Hashtag format sometimes used: #01
        ep_part = f" #{episode:02d}"
    
    # Formatting variations
    format_type = random.choices([
        "standard",    # [Studio] Title - 01 [Resolution]
        "dotted",      # [Studio].Title.-.01.(Resolution)
        "detailed",    # [Studio] Title - 01 [Resolution][Format][Extra]
        "named",       # [Studio] Title - 01 - Episode Name [Resolution]
        "minimal",     # Title - 01
    ], weights=[40, 20, 20, 15, 5], k=1)[0]
    
    # Episode title (only for "named" format or randomly for others)
    ep_title = ""
    if format_type == "named" or (random.random() < 0.2 and format_type != "dotted"):
        ep_title = random.choice(episode_titles)
    
    if format_type == "standard":
        filename = f"[{studio}] {title}{' ' + season if season else ''}{ep_part} [{resolution}].mkv"
    
    elif format_type == "dotted":
        dot_title = title.replace(' ', '.')
        dot_season = season.replace(' ', '.') if season else ""
        filename = f"[{studio}].{dot_title}{('.' + dot_season) if dot_season else ''}{ep_part.replace(' ', '.')}.({resolution}).mkv"
    
    elif format_type == "detailed":
        extras = [
            random.choice(['HEVC', 'H264', 'H.265', '10bit']), 
            random.choice(['AAC', 'FLAC', 'DualAudio']), 
            random.choice(['Multi-Sub', 'Uncensored', 'BD', 'WEB-DL'])
        ]
        extra_tags = ''.join([f"[{e}]" for e in extras if random.random() < 0.7])
        filename = f"[{studio}] {title}{' ' + season if season else ''}{ep_part} [{resolution}]{extra_tags}.mkv"
    
    elif format_type == "named":
        filename = f"[{studio}] {title}{' ' + season if season else ''}{ep_part}{f' - {ep_title}' if ep_title else ''} [{resolution}].mkv"
    
    else:  # minimal
        filename = f"{title}{ep_part}.mkv"
    
    # Label components for training
    labels = {
        "media_type": "anime",
        "title": title,
        "season": season_value,
        "episode": episode,
        "episode_title": ep_title,
        "resolution": resolution,
        "studio": studio,
    }
    
    return filename, labels

def generate_dataset(num_samples=100, balance_difficult_cases=True):
    """
    Generate a dataset of synthetic filenames with labels
    
    Args:
        num_samples: Total number of samples to generate
        balance_difficult_cases: Whether to include extra examples of challenging patterns
    """
    data = []
    
    # Calculate base distribution
    if balance_difficult_cases:
        # Balanced distribution: more tv shows and movies with years
        movie_pct = 0.35
        tv_pct = 0.35
        anime_pct = 0.2
        edge_case_pct = 0.1  # Reserve 10% for dedicated edge cases
    else:
        # Default distribution
        movie_pct = 0.33
        tv_pct = 0.33
        anime_pct = 0.34
        edge_case_pct = 0.0
    
    # Calculate number of each type to generate
    num_movies = int(num_samples * movie_pct)
    num_tv = int(num_samples * tv_pct)
    num_anime = int(num_samples * anime_pct)
    num_edge_cases = num_samples - num_movies - num_tv - num_anime  # Remainder for edge cases
    
    # Generate movies
    print(f"Generating {num_movies} movie filenames...")
    for _ in range(num_movies):
        filename, labels = generate_movie_filename()
        if filename and filename != "." and len(filename) > 5:
            data.append((filename, labels))
    
    # Generate TV shows with extra emphasis on season/episode patterns
    print(f"Generating {num_tv} TV show filenames...")
    for _ in range(num_tv):
        filename, labels = generate_tv_filename()
        if filename and filename != "." and len(filename) > 5:
            data.append((filename, labels))
    
    # Generate anime
    print(f"Generating {num_anime} anime filenames...")
    for _ in range(num_anime):
        filename, labels = generate_anime_filename()
        if filename and filename != "." and len(filename) > 5:
            data.append((filename, labels))
    
    # Generate challenging examples from edge cases
    if num_edge_cases > 0:
        print(f"Generating {num_edge_cases} targeted edge cases...")
        edge_cases = []
        
        # 1. Titles with numbers
        num_with_numbers = num_edge_cases // 4
        for _ in range(num_with_numbers):
            # Filter titles that start with numbers or contain numbers
            number_titles = [t for t in titles if any(c.isdigit() for c in t)]
            if not number_titles:
                number_titles = ["2012", "300", "21 Jump Street"]
                
            title = random.choice(number_titles)
            year = random.choice(years)
            delimiter = random.choice(delimiters)
            
            # Format ensuring title's numbers are preserved
            clean_t = clean_title(title, delimiter)
            filename = f"{clean_t}.{year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
            
            labels = {
                "media_type": "movie",
                "title": title,
                "year": year
            }
            edge_cases.append((filename, labels))
        
        # 2. Web domains in filename
        num_with_domains = num_edge_cases // 4
        for _ in range(num_with_domains):
            title = random.choice(titles)
            year = random.choice(years)
            delimiter = random.choice(delimiters)
            
            # Generate web domain pattern
            domains = ["www.UsaBit.com", "www.RARBG.to", "TorrentCounter.to", "YTS.mx"]
            domain = random.choice(domains)
            
            # Format with domain in brackets
            filename = f"[ {domain} ] - {title.replace(' ', delimiter)} ({year}) {random.choice(resolutions)}.mkv"
            
            labels = {
                "media_type": "movie",
                "title": title,
                "year": year
            }
            edge_cases.append((filename, labels))
        
        # 3. Sequels with numbers
        num_sequels = num_edge_cases // 4
        for _ in range(num_sequels):
            # Choose a title that could have a sequel
            base_titles = ["Pacific Rim", "Blade", "Star Wars", "Fast and Furious", "Spider Man", "Iron Man", "John Wick"]
            base_title = random.choice(base_titles)
            
            # Add sequel number
            sequel_number = random.randint(2, 5)
            sequel_formats = [
                f"{base_title} {sequel_number}",                      # Simple number
                f"{base_title}.{sequel_number}",                      # With dot
                f"{base_title} Chapter {sequel_number}",              # Chapter format
                f"{base_title} Part {sequel_number}",                 # Part format
                f"{base_title} {sequel_number}: Subtitle"            # With subtitle
            ]
            
            title = random.choice(sequel_formats)
            year = random.choice(years)
            delimiter = random.choice(delimiters)
            
            # Format ensuring title structure is preserved
            clean_t = title.replace(' ', delimiter)
            filename = f"{clean_t}.{year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
            
            labels = {
                "media_type": "movie",
                "title": title,
                "year": year
            }
            edge_cases.append((filename, labels))
        
        # 4. Titles with dots/periods/acronyms
        num_with_dots = num_edge_cases - len(edge_cases)
        for _ in range(num_with_dots):
            # Filter titles that contain dots or letters that look like acronyms
            acronym_titles = [t for t in titles if '.' in t or any(len(w) == 1 for w in t.split())]
            if not acronym_titles:
                acronym_titles = ["S.W.A.T", "Agents of S.H.I.E.L.D", "M.A.S.H", "G.I. Joe"]
                
            title = random.choice(acronym_titles)
            # Sometimes use a TV show format
            if random.random() < 0.5:
                # TV show with season/episode
                season = random.randint(1, 15)
                episode = random.randint(1, 24)
                
                # Format with dots between all parts
                filename = f"{title.replace(' ', '.')}.S{season:02d}E{episode:02d}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                
                labels = {
                    "media_type": "tv",
                    "title": title,
                    "season": season,
                    "episode": episode
                }
            else:
                # Movie format
                year = random.choice(years)
                
                # Format with dots between all parts
                filename = f"{title.replace(' ', '.')}.{year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                
                labels = {
                    "media_type": "movie",
                    "title": title,
                    "year": year
                }
            
            edge_cases.append((filename, labels))
        
        # Add all edge cases to the dataset
        data.extend(edge_cases)
    
    # Generate additional challenging examples if requested
    if balance_difficult_cases and len(data) < num_samples * 1.2:  # Add up to 20% more examples
        num_additional = min(int(num_samples * 0.2), num_samples - len(data))
        if num_additional > 0:
            print(f"Generating {num_additional} additional challenging examples...")
            
            for _ in range(num_additional):
                # Select a challenge type
                challenge_type = random.choice([
                    "movie_with_year",
                    "tv_with_season_episode",
                    "tv_with_episode_title",
                    "anime_with_episode",
                    "future_date_in_title",
                    "case_sensitivity",
                    "directory_paths",
                    "movie_year_in_title"
                ])
                
                if challenge_type == "movie_with_year":
                    # Ensure movie has year in the filename
                    year = random.choice(years)
                    title = random.choice(titles)
                    delimiter = random.choice(delimiters)
                    resolution = random.choice(resolutions)
                    
                    # Format with year prominently placed
                    filename = f"{title.replace(' ', delimiter)}.{year}.{resolution}.{random.choice(sources)}.{random.choice(codecs)}.mkv"
                    labels = {
                        "media_type": "movie",
                        "title": title,
                        "year": year,
                        "resolution": resolution
                    }
                    data.append((filename, labels))
                    
                elif challenge_type == "tv_with_season_episode":
                    # Create TV show with clear season/episode markers
                    title = random.choice(titles)
                    season = random.randint(1, 35)
                    episode = random.randint(1, 99)
                    delimiter = random.choice(delimiters)
                    
                    # Choose from various season/episode formats, focusing on edge cases
                    se_formats = [
                        f"S{season:02d}E{episode:02d}",             # Standard format
                        f"S{season}E{episode}",                     # No zero padding
                        f"s{season:02d}e{episode:02d}",             # Lowercase
                        f"{season}x{episode:02d}",                  # x separator
                        f"Season {season} Episode {episode}",       # Text format
                        f"Season{season}Episode{episode}",          # No spaces text format
                        f"S{season:02d} - E{episode:02d}",          # With dash between S and E
                        f"s{season}x{episode:02d}v2",               # With version suffix
                        f"S{season:02d}E{episode:02d}-E{episode+1:02d}" # Multi-episode
                    ]
                    
                    # Select a format, bias toward less common ones
                    se_format = random.choices(
                        se_formats, 
                        weights=[1, 2, 2, 3, 3, 3, 3, 3, 3],  # Higher weights for uncommon formats
                        k=1
                    )[0]
                    
                    filename = f"{title.replace(' ', delimiter)}.{se_format}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                    labels = {
                        "media_type": "tv",
                        "title": title,
                        "season": season,
                        "episode": episode
                    }
                    data.append((filename, labels))
                    
                elif challenge_type == "tv_with_episode_title":
                    # TV show with episode title
                    title = random.choice(titles)
                    season = random.randint(1, 15)
                    episode = random.randint(1, 24)
                    episode_title = random.choice(episode_titles)
                    delimiter = random.choice(delimiters)
                    
                    filename = f"{title.replace(' ', delimiter)}.S{season:02d}E{episode:02d}.{episode_title.replace(' ', delimiter)}.{random.choice(resolutions)}.mkv"
                    labels = {
                        "media_type": "tv",
                        "title": title,
                        "season": season,
                        "episode": episode,
                        "episode_title": episode_title
                    }
                    data.append((filename, labels))
                    
                elif challenge_type == "anime_with_episode":
                    # Anime with episode number
                    title = random.choice(titles)
                    episode = random.randint(1, 24)
                    studio = random.choice(anime_studios)
                    resolution = random.choice(resolutions)
                    
                    filename = f"[{studio}] {title} - {episode:02d} [{resolution}].mkv"
                    labels = {
                        "media_type": "anime",
                        "title": title,
                        "episode": episode,
                        "studio": studio
                    }
                    data.append((filename, labels))
                
                elif challenge_type == "future_date_in_title":
                    # Create movie with a future date in the title
                    title_with_date = random.choice([
                        "Blade Runner 2049", "Death Race 2000", "2012", "1984", 
                        "Space 1999", "2001 A Space Odyssey"
                    ])
                    
                    # Extract the actual year from the title
                    actual_year = random.choice(years)
                    delimiter = random.choice(delimiters)
                    
                    # Format ensuring the future date is preserved in the title
                    clean_t = clean_title(title_with_date, delimiter)
                    filename = f"{clean_t}.{actual_year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                    
                    labels = {
                        "media_type": "movie",
                        "title": title_with_date,
                        "year": actual_year
                    }
                    data.append((filename, labels))
                
                elif challenge_type == "case_sensitivity":
                    # Create filename with mixed case
                    title = random.choice(titles)
                    
                    if random.random() < 0.5:
                        # TV show with lowercase season/episode markers
                        season = random.randint(1, 15)
                        episode = random.randint(1, 24)
                        delimiter = random.choice(delimiters)
                        
                        # Use lowercase for everything
                        filename = f"{title.replace(' ', delimiter).lower()}.s{season:02d}e{episode:02d}.{random.choice(resolutions).lower()}.{random.choice(sources).lower()}.mkv"
                        
                        labels = {
                            "media_type": "tv",
                            "title": title,
                            "season": season,
                            "episode": episode
                        }
                    else:
                        # Movie with mixed case
                        year = random.choice(years)
                        delimiter = random.choice(delimiters)
                        
                        # Lowercase the title
                        filename = f"{title.replace(' ', delimiter).lower()}.{year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                        
                        labels = {
                            "media_type": "movie",
                            "title": title,
                            "year": year
                        }
                    
                    data.append((filename, labels))
                
                elif challenge_type == "directory_paths":
                    # Create filename with directory structure
                    title = random.choice(titles)
                    
                    if random.random() < 0.5:
                        # TV show with season directory
                        season = random.randint(1, 10)
                        episode = random.randint(1, 20)
                        episode_title = random.choice(episode_titles) if random.random() < 0.5 else ""
                        
                        # Create directory path
                        dir_path = f"{title} Season {season}"
                        
                        # Create filename
                        filename = f"{dir_path}/Episode {episode}{f' - {episode_title}' if episode_title else ''}"
                        
                        labels = {
                            "media_type": "tv",
                            "title": title,
                            "season": season,
                            "episode": episode,
                            "episode_title": episode_title
                        }
                    else:
                        # Anime with structure
                        episode = random.randint(1, 24)
                        
                        # Create directory path
                        dir_path = f"{title} [1080p]"
                        
                        # Create filename
                        filename = f"{dir_path}/Episode {episode}"
                        
                        labels = {
                            "media_type": "anime",
                            "title": title,
                            "episode": episode
                        }
                    
                    data.append((filename, labels))
                
                elif challenge_type == "movie_year_in_title":
                    # Create movie where the title includes a year that's not the release year
                    year_titles = ["2012", "1917", "1984", "2001 A Space Odyssey"]
                    title = random.choice(year_titles)
                    
                    # Get a different year for the actual release
                    release_year = None
                    while not release_year or str(release_year) in title:
                        release_year = random.choice(years)
                    
                    delimiter = random.choice(delimiters)
                    
                    # Format ensuring both the title's year and release year are clear
                    clean_t = clean_title(title, delimiter)
                    filename = f"{clean_t}.{release_year}.{random.choice(resolutions)}.{random.choice(sources)}.mkv"
                    
                    labels = {
                        "media_type": "movie",
                        "title": title,
                        "year": release_year
                    }
                    data.append((filename, labels))
    
    # If we still have too few examples, fill with random types
    while len(data) < num_samples:
        gen_type = random.choice(["movie", "tv", "anime"])
        
        if gen_type == "movie":
            filename, labels = generate_movie_filename()
        elif gen_type == "tv":
            filename, labels = generate_tv_filename()
        else:
            filename, labels = generate_anime_filename()
            
        if filename and filename != "." and len(filename) > 5:
            data.append((filename, labels))
    
    # Shuffle the data
    random.shuffle(data)
    
    # Trim to exactly num_samples
    return data[:num_samples]

def write_to_csv(data, filename="synthetic_media_data.csv"):
    """Write the dataset to a CSV file"""
    all_keys = set()
    for _, labels in data:
        all_keys.update(labels.keys())
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['filename'] + sorted(list(all_keys))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for filename, labels in data:
            row = {'filename': filename}
            row.update(labels)
            writer.writerow(row)
    
    print(f"Dataset written to {filename}")

if __name__ == "__main__":
    # Generate 5000 synthetic examples with balanced difficult cases
    print("Generating synthetic media filename dataset...")
    dataset = generate_dataset(5000, balance_difficult_cases=True)
    write_to_csv(dataset)
    
    print("\nSample filenames:")
    for i, (filename, _) in enumerate(dataset[:10]):
        print(f"{i+1}. {filename}")
    
    # Print some statistics
    media_types = {}
    field_counts = {field: 0 for field in ["title", "year", "season", "episode", "episode_title"]}
    
    for _, labels in dataset:
        media_type = labels.get("media_type", "unknown")
        media_types[media_type] = media_types.get(media_type, 0) + 1
        
        for field in field_counts:
            if field in labels and labels[field]:
                field_counts[field] += 1
    
    print("\nDataset statistics:")
    print(f"Total examples: {len(dataset)}")
    for media_type, count in media_types.items():
        print(f"  {media_type}: {count} ({count/len(dataset)*100:.1f}%)")
    
    print("\nField coverage:")
    for field, count in field_counts.items():
        print(f"  {field}: {count} ({count/len(dataset)*100:.1f}%)")