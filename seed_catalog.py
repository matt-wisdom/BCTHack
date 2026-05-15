import sys
import os
import csv
import glob
import zipfile

# Ensure the app module can be imported
sys.path.append(os.getcwd())
from app.memory import memory_manager

def unzip_datasets(dataset_dir):
    print(f"📦 Checking for zipped datasets in {dataset_dir}...")
    zip_files = glob.glob(os.path.join(dataset_dir, "*.zip"))
    for zip_path in zip_files:
        print(f"  ... unzipping {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    if zip_files:
        print(f"✅ Unzipped {len(zip_files)} files.\n")
    else:
        print("  ... no zip files found.\n")

def seed_dataset(file_path, source_name, row_limit=None):
    print(f"📂 Processing {source_name} dataset: {file_path}...")
    collection = memory_manager.product_collection
    
    # Try UTF-8 first, fallback to latin-1 if needed
    try:
        f = open(file_path, mode='r', encoding='utf-8')
        reader = csv.DictReader(f)
        # Trigger a read to check encoding
        _ = reader.fieldnames
    except UnicodeDecodeError:
        f.close()
        print(f"  ... UTF-8 failed, retrying with latin-1")
        f = open(file_path, mode='r', encoding='latin-1')
        reader = csv.DictReader(f)

    try:
        batch_docs = []
        batch_metas = []
        batch_ids = []
        
        for i, row in enumerate(reader):
            if row_limit and i >= row_limit:
                break
                
            # Generic content extraction based on known headers
            if source_name == "amazon":
                title = row.get("title", "Unknown Amazon Product")
                price = row.get("price", "N/A")
                stars = row.get("stars", "N/A")
                content = f"Source: Amazon\nProduct: {title}\nPrice: {price} USD\nRating: {stars} stars"
            elif source_name == "jumia":
                name = row.get("product_name", "Unknown Jumia Product")
                price = row.get("price", "N/A")
                rating = row.get("avg_rate", "N/A")
                content = f"Source: Jumia\nProduct: {name}\nPrice: {price}\nRating: {rating}"
            elif source_name == "store":
                name = row.get("Name", "Unknown Local Product")
                price = row.get("Current Price ₦", "N/A")
                rating = row.get("Rating", "N/A")
                content = f"Source: Local Store\nProduct: {name}\nPrice: {price} NGN\nRating: {rating} stars"
            else:
                content = str(row)

            batch_docs.append(content)
            batch_metas.append({"dataset_source": source_name, "original_index": i})
            batch_ids.append(f"{source_name}_{i}")
            
            # Chroma works better with batches
            if len(batch_docs) >= 100:
                collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                batch_docs, batch_metas, batch_ids = [], [], []
                print(f"  ... inserted {i+1} rows")

        # Final batch
        if batch_docs:
            collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
    finally:
        f.close()
    
    print(f"✅ Finished {source_name} dataset.\n")

def run_seeder():
    print("🚀 Initializing system for seeding...")
    # Path to dataset folder
    dataset_dir = "dataset"
    
    # Ensure all files are unzipped
    unzip_datasets(dataset_dir)
    
    memory_manager.initialize()
    
    # Map filenames to source names
    mappings = {
        "amazon_products.csv": "amazon",
        "jumia_products.csv": "jumia",
        "store_df_multi_page.csv": "store"
    }
    
    for filename, source in mappings.items():
        file_path = os.path.join(dataset_dir, filename)
        if os.path.exists(file_path):
            # Limit amazon to 1000 rows for speed
            limit = 1000 if source == "amazon" else None
            seed_dataset(file_path, source, row_limit=limit)
        else:
            print(f"⚠️ Warning: {file_path} not found. Skipping.")

if __name__ == "__main__":
    run_seeder()
    print("🎉 All datasets seeded successfully!")
