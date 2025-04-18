#!/usr/bin/env python3
import random
import csv
import re
from datetime import datetime, timedelta

# Sample data for filling templates
titles = [
    "The Shawshank Redemption", "The Godfather", "Pulp Fiction", "The Dark Knight", 
    "Fight Club", "Inception", "Interstellar", "Parasite", "Avengers Endgame", 
    "The Matrix", "Breaking Bad", "Game of Thrones", "Stranger Things", "The Office", 
    "Friends", "The Mandalorian", "Attack on Titan", "Demon Slayer", "My Hero Academia",
    "Jujutsu Kaisen", "One Piece", "Naruto", "Death Note", "Fullmetal Alchemist"
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
    """Generate a synthetic TV show filename"""
    title = random.choice(titles)
    season = random.randint(1, 15)
    episode = random.randint(1, 24)
    year = random.choice(years)
    delimiter = random.choice(delimiters)
    resolution = random.choice(resolutions)
    source = random.choice(sources)
    codec = random.choice(codecs)
    group = random.choice(release_groups)
    
    # Clean the title
    clean_t = clean_title(title, delimiter)
    
    # Episode format
    ep_format = random.choice([
        f"S{season:02d}E{episode:02d}",
        f"S{season:02d}E{episode:02d}{random.choice(['-', 'E'])}{episode+1:02d}",  # Multi-episode
        f"{season}x{episode:02d}",
        f"{year}.{random.randint(1,12):02d}.{random.randint(1,28):02d}"  # Date format
    ])
    
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
    
    # Label components for training
    labels = {
        "media_type": "tv",
        "title": title,
        "season": season if "S" in ep_format or "x" in ep_format else "",
        "episode": episode if "E" in ep_format or "x" in ep_format else "",
        "episode_title": random.choice(episode_titles) if ep_title else "",
        "resolution": resolution,
        "source": source if source in components else "",
    }
    
    return filename, labels

def generate_anime_filename():
    """Generate a synthetic anime filename"""
    title = random.choice(titles)
    episode = random.randint(1, 24)
    season = random.choice(["", "S2", "S3"]) if random.random() < 0.3 else ""
    resolution = random.choice(resolutions)
    studio = random.choice(anime_studios)
    
    # Format variations
    formats = [
        f"[{studio}] {title}{' ' + season if season else ''} - {episode:02d} [{resolution}].mkv",
        f"[{studio}] {title.replace(' ', '.')}{('.' + season) if season else ''}.-.{episode:02d}.({resolution}).mkv",
        f"[{studio}] {title} - {episode:02d} [{resolution}][{random.choice(['HEVC', '10bit', 'Multiple Subtitle'])}].mkv",
        f"[{studio}] {title}{' ' + season if season else ''} - {episode:02d} [BD][{resolution}][{random.choice(['HEVC', 'DualAudio', 'Uncensored'])}].mkv"
    ]
    
    filename = random.choice(formats)
    
    # Label components for training
    labels = {
        "media_type": "anime",
        "title": title,
        "season": season,
        "episode": episode,
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
        movie_pct = 0.4
        tv_pct = 0.4
        anime_pct = 0.2
    else:
        # Default distribution
        movie_pct = 0.33
        tv_pct = 0.33
        anime_pct = 0.34
    
    # Calculate number of each type to generate
    num_movies = int(num_samples * movie_pct)
    num_tv = int(num_samples * tv_pct)
    num_anime = num_samples - num_movies - num_tv  # Remainder to ensure total = num_samples
    
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
    
    # Generate additional challenging examples if requested
    if balance_difficult_cases and len(data) < num_samples * 1.2:  # Add up to 20% more examples
        num_additional = int(num_samples * 0.2)
        print(f"Generating {num_additional} additional challenging examples...")
        
        for _ in range(num_additional):
            # Select a challenge type
            challenge_type = random.choice([
                "movie_with_year",
                "tv_with_season_episode",
                "tv_with_episode_title",
                "anime_with_episode"
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
                season = random.randint(1, 15)
                episode = random.randint(1, 24)
                delimiter = random.choice(delimiters)
                
                # Choose from various season/episode formats
                se_format = random.choice([
                    f"S{season:02d}E{episode:02d}",
                    f"S{season:02d}E{episode}",
                    f"s{season:02d}e{episode:02d}",
                    f"{season}x{episode:02d}"
                ])
                
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