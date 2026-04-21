import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber

def split_documents(file_paths_with_year):
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    for file_path, year in file_paths_with_year:
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()
        chunks = text_splitter.split_documents(raw_documents)
        
        # 메타데이터 부여
        for chunk in chunks:
            chunk.metadata["source_year"] = year

        all_chunks.extend(chunks)
    
    print(f"Total Combined Chunks: {len(chunks)}")
    return all_chunks


def export_chunks_to_jsonl(chunks, output_path="chunks/chunks.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            content = chunk.page_content.replace("\n", " ")
            source_year = chunk.metadata.get("source_year", None)

            record = {
                "id": idx,
                "chunk": content,
                "source_year": source_year
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"총 {len(chunks)}개의 청크가 {output_path}에 저장되었습니다.")

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # print(f)
        for line in f:
            try:
                if line.strip():
                    data.append(json.loads(line))
            except:
                print(line)

    return data

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_prompt(path="prompts/prompts.json"):
    with open(path, "R", encoding="utf-8") as f:
        prompts = json.load(f)

    return prompts


def save_data(file_path, dataset):

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def save_prompt(prompts, path="prompts/prompts.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)