import re

def chunk_text(text, max_chunk_size=1000, min_chunk_size=200):
    if not text:
        return []

    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        if current_length + len(para) <= max_chunk_size:
            current_chunk.append(para)
            current_length += len(para) + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            if len(para) > max_chunk_size:
                sentences = sentence_endings.split(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if current_length + len(sentence) <= max_chunk_size:
                        current_chunk.append(sentence)
                        current_length += len(sentence) + 1 
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
            else:
                current_chunk = [para]
                current_length = len(para)
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk) if "\n\n" in text else " ".join(current_chunk))
        
    final_chunks = []
    if not chunks:
        return []
        
    temp_chunk = chunks[0]
    for i in range(1, len(chunks)):
        if len(temp_chunk) < min_chunk_size:
            temp_chunk = temp_chunk + "\n\n" + chunks[i]
        else:
            final_chunks.append(temp_chunk)
            temp_chunk = chunks[i]
    final_chunks.append(temp_chunk)
    
    return final_chunks
