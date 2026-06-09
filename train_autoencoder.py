import argparse
from itertools import islice
from pathlib import Path

import joblib
from datasets import load_dataset
from tqdm import tqdm
from turftopic.late import LateSentenceTransformer, flatten_repr

from utils.autoencoder import TopKAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser(prog="train_autoencoder")
    parser.add_argument("encoder_name")
    return parser.parse_args()


# Roughly corresponds to .5B tokens
N_BATCHES = 500


def main(encoder_name: str):
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    encoder = LateSentenceTransformer(encoder_name)
    autoencoder = TopKAutoEncoder(
        n_latent=10_000,
        top_k=15,
        lr=1e-3,
        alpha=0.03,
        batch_size=4096,
        n_epochs=1,
        random_state=42,
        show_progress_bar=False,
    )
    for i_epoch in range(3):
        batches = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        ).batch(batch_size=1000)
        batches = islice(batches, N_BATCHES)
        batches = tqdm(
            batches,
            desc=f"Going through batches for epoch {i_epoch}",
            total=N_BATCHES,
        )
        for i_batch, batch in enumerate(batches):
            token_embeddings, offsets = encoder.encode_tokens(
                list(batch["text"]), show_progress_bar=False
            )
            flat_token_embeddings, lengths = flatten_repr(token_embeddings)
            autoencoder.partial_fit(flat_token_embeddings, n_epochs=1)
            if i_batch % (N_BATCHES // 10):
                print("Saving checkpoint...")
                joblib.dump(
                    autoencoder,
                    checkpoint_dir.joinpath(f"epoch_{i_epoch}-batch_{i_batch}.joblib"),
                )
    print("Saving final autoencoder.")
    joblib.dump(
        autoencoder,
        checkpoint_dir.joinpath("final.joblib"),
    )
    print("DONE")


if __name__ == "__main__":
    args = parse_args()
    main(args.encoder_name)
