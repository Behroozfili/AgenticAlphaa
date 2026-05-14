import os

def create_project_structure():
    # Set to current directory since you've already created the main folder
    base_dir = "."

    # Define the directory mapping and their respective files
    structure = {
        "agents": ["manager_agent.py", "research_agent.py", "financial_agent.py", "sentiment_agent.py"],
        "rag": ["vector_store.py", "graph_store.py", "hybrid_retriever.py", "document_ingestion.py"],
        "tools": ["yahoo_finance.py", "sec_edgar.py", "tavily_search.py", "reddit_scraper.py", "sentiment_scorer.py"],
        "memory": ["mem0_client.py"],
        "evaluation": ["backtester.py", "metrics.py"],
        "evaluation/reports": [],  # Directory for backtest logs
        "api": ["main.py", "schemas.py"],
        "api/routes": ["analyze.py", "history.py"],
        "frontend": [],
        "scheduler": ["daily_refresh.py"],
        "tests": [],
        ".github/workflows": ["deploy.yml", "daily_refresh.yml"]
    }

    # Files located in the root directory
    root_files = [".env.example", ".gitignore", "docker-compose.yml"]

    print(f"🚀 Initializing structure in: {os.getcwd()}")

    # Generate sub-directories and placeholder files
    for folder, files in structure.items():
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            # Only create the file if it doesn't already exist to avoid overwriting
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Alpha-Agent Node: {file}\n")
                print(f"  └─ Created file: {folder}/{file}")
            else:
                print(f"  ⚠ Skipping: {folder}/{file} (Already exists)")

    # Generate root-level configuration files
    for r_file in root_files:
        r_file_path = os.path.join(base_dir, r_file)
        if not os.path.exists(r_file_path):
            with open(r_file_path, 'w', encoding='utf-8') as f:
                if r_file == ".gitignore":
                    f.write(".env\n__pycache__/\n*.pyc\nvenv/\n.vscode/\n")
            print(f"✔ Created root file: {r_file}")

    print("\n✅ Project skeleton is ready!")

if __name__ == "__main__":
    create_project_structure()
    