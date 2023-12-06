def subsampling(vocabulary, training_words, next_random) :
    to_remove = []
    if len(training_words) > 0 : 
        for word in vocabulary : 
            if vocabulary[word] == 0 :
                to_remove.append(word)
                continue 
            f_x = vocabulary[word]/len(training_words)
            ran = (pow(f_x, 0.5) + 1)/f_x
            next_random = next_random * 25214903917 + 11
            if ran < next_random : 
                continue 
            else :
                to_remove.append(word)
                
        return to_remove
    else : 
        return -1 