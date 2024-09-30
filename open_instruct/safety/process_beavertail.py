import json
from datasets import load_dataset
from tqdm import tqdm


def process_pku_saferlhf(output_file):
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

    with open(output_file, "w") as outfile:
        for split in dataset.keys():
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                if item["safer_response_id"] == '0':  # Changed to string '0'
                    chosen = item["response_0"]
                    rejected = item["response_1"]
                else:
                    chosen = item["response_1"]
                    rejected = item["response_0"]

                prompt = item["prompt"]

                preference_data = {
                    "chosen": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen}
                    ],
                    "rejected": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected}
                    ],
                    "source": "PKU-SafeRLHF"
                }

                json.dump(preference_data, outfile)
                outfile.write('\n')  # Add newline for JSONL format

    print("Processing done!")
    print(f"Data saved in {output_file}")


if __name__ == "__main__":
    output_file = "/net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/open_instruct/safety/pku_saferlhf_preferences.jsonl"
    process_pku_saferlhf(output_file)