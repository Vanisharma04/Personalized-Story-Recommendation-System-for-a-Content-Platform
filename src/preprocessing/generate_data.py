"""Synthetic story-platform dataset generator.

Simulates a Pratilipi/Wattpad-style platform with realistic structure:

* Stories have genre-specific titles/descriptions, a latent quality score,
  and power-law popularity.
* Users have 1-3 preferred genres (Dirichlet affinity vector) and an
  activity level (power-law: a few heavy readers, many casual ones).
* Interactions are sampled from user affinity x story quality, with
  engagement signals (reading_time, completion_rate, likes) correlated
  with how well the story matches the user's taste.

This gives downstream models real signal to learn instead of noise.
"""
import numpy as np
import pandas as pd

from src import config
from src.utils.io import setup_logger

log = setup_logger("data-gen")

# ------------------------------------------------------------------ #
# Genre-specific vocabulary for titles & descriptions                 #
# ------------------------------------------------------------------ #
GENRE_LEXICON = {
    "Fantasy": {
        "subjects": ["The Lost Kingdom", "The Dragon's Heir", "The Shadow Mage",
                     "The Crystal Throne", "The Last Enchanter", "The Moonlit Crown",
                     "The Forgotten Realm", "The Ember Witch", "The Silver Citadel",
                     "The Runebound Oath"],
        "themes": ["an ancient prophecy", "a forbidden spell", "a dying magical realm",
                   "a war between old gods", "a cursed bloodline", "a hidden portal"],
        "actors": ["a reluctant young mage", "an exiled princess", "a dragon rider",
                   "an orphaned thief with strange powers", "a fallen knight"],
    },
    "Romance": {
        "subjects": ["Letters to Meera", "The Wedding Planner's Secret", "Monsoon Hearts",
                     "Coffee at Midnight", "The Arranged Truth", "Second Chance Summer",
                     "His Last Love Letter", "The Bookshop Romance", "Paper Roses",
                     "Tides of Longing"],
        "themes": ["a love that defies family expectations", "a second chance at romance",
                   "an arranged match that becomes real", "a long-distance promise",
                   "a secret admirer's confession", "a friendship turning into love"],
        "actors": ["a small-town teacher", "an ambitious chef", "a widowed architect",
                   "a shy librarian", "a struggling musician"],
    },
    "Mystery": {
        "subjects": ["The Silent Witness", "Murder at the Haveli", "The Vanishing Hour",
                     "The Locked Room", "The Cartographer's Code", "A Study in Shadows",
                     "The Midnight Caller", "The Seventh Clue", "The Glass Alibi",
                     "Echoes in the Archive"],
        "themes": ["a decades-old cold case", "a disappearance nobody reported",
                   "a murder staged as suicide", "a coded diary left behind",
                   "an inheritance with deadly strings", "a witness who lies"],
        "actors": ["a retired detective", "a sharp-eyed journalist", "an amateur sleuth",
                   "a forensic accountant", "a librarian who notices everything"],
    },
    "Science Fiction": {
        "subjects": ["The Mars Protocol", "Signal from Kepler", "The Memory Merchants",
                     "Quantum Drift", "The Last Uplink", "Children of the Dyson Ring",
                     "The Synthetic Heart", "Orbital Decay", "The Terraform Wars",
                     "Ghost in the Lattice"],
        "themes": ["first contact gone wrong", "a colony ship losing its memory",
                   "an AI questioning its purpose", "time dilation tearing a family apart",
                   "a black-market for memories", "terraforming that awakens something"],
        "actors": ["a disgraced astrophysicist", "a rogue android", "a deep-space salvager",
                   "a quantum engineer", "the last human translator"],
    },
    "Horror": {
        "subjects": ["The House on Mill Road", "Whispers in the Walls", "The Night Shift",
                     "Don't Open the Cellar", "The Hollow Children", "Static",
                     "The Mirror Room", "Feed the Well", "The Smiling Man",
                     "What the Forest Keeps"],
        "themes": ["a house that remembers its dead", "a ritual that must not be finished",
                   "a town that vanishes at night", "an entity that mimics loved ones",
                   "a curse passed through photographs", "something living under the floor"],
        "actors": ["a night-shift nurse", "a skeptical paranormal blogger",
                   "a grieving father", "a new tenant", "a rural schoolteacher"],
    },
    "Historical Fiction": {
        "subjects": ["The Salt March Diaries", "Daughters of the Empire", "The Silk Route",
                     "A Court of Ashes", "The Mughal Cartographer", "Letters from 1947",
                     "The Spice Merchant's Wife", "Banners of the Deccan",
                     "The Astronomer of Ujjain", "Shadows of Partition"],
        "themes": ["a family divided by partition", "a forbidden craft in a royal court",
                   "a merchant caravan crossing empires", "a rebellion brewing in silence",
                   "a love story against the freedom struggle", "a court intrigue over succession"],
        "actors": ["a court astronomer", "a rebel poet", "a merchant's daughter",
                   "a young cartographer", "a palace scribe"],
    },
    "Thriller": {
        "subjects": ["The Extraction", "Zero Hour Mumbai", "The Informant",
                     "Blackout Protocol", "The Courier", "Deadline",
                     "The Hostage Negotiator", "Crossfire", "The Sleeper Cell",
                     "Forty-Eight Hours"],
        "themes": ["a conspiracy reaching the highest office", "a hostage swap going sideways",
                   "an informant burned by their own agency", "a city-wide blackout hiding a heist",
                   "a journalist with one night to publish", "a double agent's final job"],
        "actors": ["an ex-intelligence officer", "a cyber-crime analyst",
                   "an investigative journalist", "a bomb-disposal expert", "a getaway driver"],
    },
    "Adventure": {
        "subjects": ["The River of Gold", "Summit Fever", "The Lost Expedition",
                     "Compass of Storms", "The Treasure of Hampi", "Beyond the Ice Wall",
                     "The Jungle Codex", "Sailing the Forgotten Sea", "The Desert Crossing",
                     "Peak of No Return"],
        "themes": ["a treasure map with a missing corner", "an expedition that never returned",
                   "a race across a dying glacier", "a shipwreck holding an old secret",
                   "a jungle hiding a lost city", "a desert crossing against time"],
        "actors": ["a washed-up mountaineer", "a treasure-hunting historian",
                   "a river guide", "a teenage stowaway", "a wildlife photographer"],
    },
    "Drama": {
        "subjects": ["The Inheritance", "What We Leave Behind", "The Family Table",
                     "Half a Life", "The Apology", "Sons of the Soil",
                     "The Long Way Home", "Unspoken", "The Visiting Hour",
                     "A House Divided"],
        "themes": ["a family secret surfacing at a funeral", "two brothers and one farm",
                   "a mother's choice that echoes for decades", "an apology thirty years late",
                   "a homecoming that reopens old wounds", "a will that splits a family"],
        "actors": ["an estranged son", "a retired schoolmistress", "a first-generation doctor",
                   "a single mother", "an aging patriarch"],
    },
    "Mythology": {
        "subjects": ["The Curse of Yayati", "Daughter of the Ocean", "The Fifth Veda",
                     "Ashwatthama's Burden", "The Churning", "Songs of the Naga",
                     "The Forest Exile", "Karna's Vow", "The Celestial Wager",
                     "Rise of the Asura"],
        "themes": ["a forgotten vow binding the gods", "a mortal chosen for a divine task",
                   "a curse traded between generations", "a retelling from the vanquished side",
                   "a celestial weapon hidden on earth", "a sage's prophecy misread"],
        "actors": ["a cursed warrior", "a river goddess in mortal form", "a temple sculptor",
                   "an exiled prince", "a storyteller who remembers past lives"],
    },
}

AUTHOR_FIRST = ["Aarav", "Vani", "Ishita", "Rohan", "Meera", "Kabir", "Ananya", "Dev",
                "Priya", "Arjun", "Sara", "Vikram", "Naina", "Aditya", "Zoya", "Rahul",
                "Elena", "Marcus", "Lin", "Sofia"]
AUTHOR_LAST = ["Sharma", "Iyer", "Khan", "Mehta", "Das", "Kapoor", "Reddy", "Bose",
               "Nair", "Joshi", "Petrov", "Tanaka", "Okafor", "Alvarez", "Fernandes",
               "Chatterjee", "Menon", "Gupta", "Rao", "Singh"]

DESC_OPENERS = ["When", "After", "In a world where", "Haunted by", "Driven by", "Against"]
DESC_CLOSERS = [
    "Nothing will ever be the same again.",
    "But every secret has its price.",
    "And time is running out.",
    "What follows changes everything.",
    "Some doors, once opened, never close.",
    "The truth is closer than anyone dares to admit.",
]


def _make_stories(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for sid in range(config.N_STORIES):
        genre = config.GENRES[sid % len(config.GENRES)]
        lex = GENRE_LEXICON[genre]
        subject = rng.choice(lex["subjects"])
        theme = rng.choice(lex["themes"])
        actor = rng.choice(lex["actors"])
        # Disambiguate repeated subjects with a sequel-style suffix.
        suffix = rng.choice(["", "", "", " II", ": Origins", ": The Reckoning",
                             " Reborn", ": Legacy"])
        title = f"{subject}{suffix}"
        description = (
            f"{rng.choice(DESC_OPENERS)} {actor} is drawn into {theme}, "
            f"they must confront what it truly costs. {rng.choice(DESC_CLOSERS)}"
        )
        rows.append({
            "story_id": f"S{sid:05d}",
            "title": title,
            "description": description,
            "genre": genre,
            "author": f"{rng.choice(AUTHOR_FIRST)} {rng.choice(AUTHOR_LAST)}",
            "avg_reading_minutes": float(np.round(rng.uniform(4, 45), 1)),
            # latent quality drives engagement; never exposed as a feature directly
            "quality": float(np.clip(rng.normal(0.6, 0.18), 0.05, 0.98)),
            "publish_days_ago": int(rng.integers(1, 720)),
        })
    return pd.DataFrame(rows)


def _make_users(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n_genres = len(config.GENRES)
    for uid in range(config.N_USERS):
        # Sparse Dirichlet -> 1-3 dominant genres per user
        affinity = rng.dirichlet(np.full(n_genres, 0.25))
        preferred = config.GENRES[int(np.argmax(affinity))]
        rows.append({
            "user_id": f"U{uid:05d}",
            "age": int(np.clip(rng.normal(28, 9), 13, 70)),
            "preferred_genre": preferred,
            "affinity": affinity,
            # power-law activity: most users casual, a few voracious
            "activity": float(np.clip(rng.pareto(1.6) + 0.3, 0.3, 25.0)),
        })
    return pd.DataFrame(rows)


def _make_interactions(rng: np.random.Generator, users: pd.DataFrame,
                       stories: pd.DataFrame) -> pd.DataFrame:
    genre_to_idx = {g: i for i, g in enumerate(config.GENRES)}
    story_genre_idx = stories["genre"].map(genre_to_idx).to_numpy()
    story_quality = stories["quality"].to_numpy()
    story_minutes = stories["avg_reading_minutes"].to_numpy()
    story_ids = stories["story_id"].to_numpy()
    n_stories = len(stories)

    # power-law base popularity independent of quality (exposure bias)
    popularity = rng.pareto(1.2, n_stories) + 1.0
    popularity /= popularity.sum()

    activity = users["activity"].to_numpy()
    n_per_user = np.maximum(
        5,
        (activity / activity.sum() * config.TARGET_INTERACTIONS).astype(int),
    )

    end_ts = pd.Timestamp("2026-06-01")
    all_rows = []
    for u_idx, user in enumerate(users.itertuples(index=False)):
        affinity = np.asarray(user.affinity)
        # P(read) ~ genre affinity * quality * exposure popularity
        score = affinity[story_genre_idx] * (0.4 + story_quality) * popularity
        prob = score / score.sum()
        n = min(int(n_per_user[u_idx]), n_stories - 1)
        chosen = rng.choice(n_stories, size=n, replace=False, p=prob)

        # chronological timestamps over the past year, denser recently
        days_back = np.sort(rng.exponential(90, n))[::-1].clip(0, 365)
        for k, s_idx in enumerate(chosen):
            match = affinity[story_genre_idx[s_idx]] / (affinity.max() + 1e-9)
            base = 0.55 * match + 0.45 * story_quality[s_idx]
            completion = float(np.clip(rng.normal(base, 0.15), 0.02, 1.0))
            reading_time = float(np.round(
                story_minutes[s_idx] * completion * rng.uniform(0.8, 1.2), 1))
            liked = int(rng.random() < (base ** 2) * 0.9)
            ts = end_ts - pd.Timedelta(days=float(days_back[k]),
                                       minutes=float(rng.integers(0, 1440)))
            all_rows.append({
                "user_id": user.user_id,
                "story_id": story_ids[s_idx],
                "clicks": int(1 + rng.poisson(0.3)),
                "reading_time": reading_time,
                "likes": liked,
                "completion_rate": round(completion, 3),
                "timestamp": ts,
            })

    df = pd.DataFrame(all_rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def main() -> None:
    rng = np.random.default_rng(config.SEED)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    stories = _make_stories(rng)
    users = _make_users(rng)
    interactions = _make_interactions(rng, users, stories)

    stories.to_csv(config.RAW_DIR / "stories.csv", index=False)
    users.drop(columns=["affinity"]).to_csv(config.RAW_DIR / "users.csv", index=False)
    interactions.to_csv(config.RAW_DIR / "interactions.csv", index=False)

    log.info("stories: %d | users: %d | interactions: %d",
             len(stories), len(users), len(interactions))
    log.info("saved raw dataset to %s", config.RAW_DIR)


if __name__ == "__main__":
    main()
