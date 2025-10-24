def generate_translation(model: MachineTranslation, 
                         tokenizer: AutoTokenizer, 
                         src_text: str, 
                         device: str, 
                         max_len: int = 50, 
                         method: str = 'greedy'):
    """
    Generates a translation for a given source text using the new model.
    """
    model.eval()
    
    # 1. Tokenize the source text
    src_tokenized = tokenizer(src_text, truncation=True, max_length=32, return_tensors="pt")
    src_tensor = src_tokenized['input_ids'].to(device)
    
    tgt_tokens = [tokenizer.bos_token_id]
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            
            # 4. Forward pass
            logits = model(model(src_tensor, tgt_tensor))
            # Logits shape: [1, current_tgt_len, vocab_size]
            
            # 5. Get the logits for the very last token
            next_token_logits = logits[:, -1, :] # Shape: [1, vocab_size]
            
            # 6. Select the next token
            if method == 'greedy':
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            elif method == 'sampling':
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                raise ValueError("Method must be 'greedy' or 'sampling'")

            # 7. Append the predicted token
            tgt_tokens.append(next_token_id)
            
            # 8. Stop if the EOS (End-Of-Sequence) token is generated
            if next_token_id == tokenizer.eos_token_id:
                break

    # 9. Decode the generated token IDs into a string
    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation


def generate_translation_beam_search(model: MachineTranslation, 
                                     tokenizer: AutoTokenizer, 
                                     src_text: str, 
                                     device: str, 
                                     max_len: int = 50, 
                                     beam_size: int = 3):
    """
    Generates a translation using beam search with the new model.
    """
    model.eval()

    # 1. Tokenize source text
    src_tokenized = tokenizer(src_text, return_tensors="pt")
    src_tensor = src_tokenized['input_ids'].to(device)

    # 2. Initialize beams: each item is a tuple of (sequence_of_tokens, score)
    initial_beam = ([tokenizer.bos_token_id], 0.0)
    beams = [initial_beam]
    
    with torch.no_grad():
        for step in range(max_len):
            all_candidates = []
            
            # 3. Expand each beam
            for sequence, score in beams:
                # If a beam has ended, add it to candidates and continue
                if sequence[-1] == tokenizer.eos_token_id:
                    all_candidates.append((sequence, score))
                    continue
                
                tgt_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

                logits = model(src_tensor, tgt_tensor)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # 4. Get top-k next tokens for the current beam
                top_k_probs, top_k_tokens = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    token_id = top_k_tokens[0, i].item()
                    token_score = top_k_probs[0, i].item()
                    
                    new_sequence = sequence + [token_id]
                    # Normalize score by sequence length to favor shorter, correct sentences
                    new_score = (score * len(sequence) + token_score) / len(new_sequence)
                    all_candidates.append((new_sequence, new_score))
            
            # 5. Prune the beams
            # Sort all candidates by their score and select the top `beam_size`
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beams = ordered[:beam_size]
            
            # Stop if all top beams have ended
            if all(b[0][-1] == tokenizer.eos_token_id for b in beams):
                break

    # 6. Select the best sequence from the final beams
    best_sequence, best_score = beams[0]
    # 7. Decode the best sequence
    translation = tokenizer.decode(best_sequence, skip_special_tokens=True)
    return translation