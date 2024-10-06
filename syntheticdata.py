
import pandas as pd
import random
import time
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer


# Step 1: Load and Clean the Dataset
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    # Remove duplicates and NaN values
    cleaned_data = data.dropna().drop_duplicates()
    print(f"Cleaned Data Shape: {cleaned_data.shape}")

    return cleaned_data


# Step 2: Initialize GPT-2 Model for Text Generation
def initialize_gpt2_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator


# Step 3: Generate Synthetic Reviews in a Structured Format
def generate_synthetic_reviews(generator, prompts, max_length=50,
                               num_sequences=5):
    synthetic_reviews = []

    for prompt in prompts:
        generated_texts = generator(prompt, max_length=max_length,
                                    num_return_sequences=num_sequences)

        for text in generated_texts:
            synthetic_reviews.append(text['generated_text'])
    return synthetic_reviews


# Step 3: Generate the Required Metadata and Assemble the Data
def create_synthetic_dataset(reviews, num_samples):
    synthetic_data = []
    for i in range(num_samples):
        review_text = reviews[i]

        # Generate random metadata for the synthetic review
        rating = random.randint(1, 5)
        title = review_text.split(".")[0] if "." in review_text else "No Title"
        asin = f"B000{random.randint(1000, 9999)}"
        parent_asin = asin
        user_id = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(random.randint(1577836800, 1609459200)))  # Random timestamp between 2020 and 2021
        helpful_vote = random.randint(0, 10)
        verified_purchase = random.choice([True, False])
        date, review_time = timestamp.split(" ")

        synthetic_data.append({
            "rating": rating,
            "title": title,
            "text": review_text,
            "asin": asin,
            "parent_asin": parent_asin,
            "user_id": user_id,
            "timestamp": timestamp,
            "helpful_vote": helpful_vote,
            "verified_purchase": verified_purchase,
            "date": date,
            "time": review_time
        })

    return pd.DataFrame(synthetic_data)


# Step 5: Save the Synthetic Dataset to a CSV File
def save_synthetic_reviews(synthetic_df, output_file):
    synthetic_df.to_csv(output_file, index=False)
    print(f"Synthetic reviews saved to {output_file}")


# Main Execution Function
def main():

    # Load and clean the original dataset
    cleaned_data = load_and_clean_data('assignment_reviews_metadata/reviews_supplements.csv')  # Replace with actual file path

    # Initialize the GPT-2 model for text generation
    generator = initialize_gpt2_model()

    # Define prompts for generating synthetic reviews (based on the dataset's common themes)
    prompts = [
        "I loved this supplement because",
        "The vitamin didn't work for me because",
        "The delivery was quick, but",
        "Great packaging, but",
        "The product was exactly as described"
    ]

    # Generate synthetic reviews
    synthetic_reviews = generate_synthetic_reviews(generator, prompts)

    # Create the synthetic dataset with structured metadata
    synthetic_df = create_synthetic_dataset(synthetic_reviews,
                                            len(synthetic_reviews))

    # Save the synthetic reviews to a CSV file
    save_synthetic_reviews(synthetic_df, 'synthetic_review_data.csv')


# Execute the script
if __name__ == '__main__':
    main()
