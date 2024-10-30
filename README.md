# Tracking-Inspiration Project

This project is a fork of the Standardized Project Gutenberg Corpus (SPGC), aiming to analyze inspiration patterns in language models using the SPGC dataset. 

## Data Download Instructions

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/tracking-inspiration.git
cd tracking-inspiration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Project Gutenberg data:
```bash
python get_data.py
```

4. Process the downloaded data:
```bash
python process_data.py
```

Note: The data download will only fetch missing files if you already have some data, making it easy to keep your dataset up-to-date by running `get_data.py` periodically.