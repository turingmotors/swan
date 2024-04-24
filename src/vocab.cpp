#include "vocab.hpp"

#include <iostream>

namespace swan {

// ResizeVocab resizes the vocab to the given size.
void ResizeVocab(Vocab& vocab, int vocab_size) {
  vocab.dict.resize(vocab_size);
}

// LoadVocab loads the vocab from the given file.
void LoadVocab(Vocab& vocab, std::ifstream& fs) {
  for (size_t i = 0; i < vocab.dict.size(); i++) {
    int len;
    vocab.dict.at(i) = "";
    fs.read((char*)&len, sizeof(int));
    for (int j = 0; j < len; ++j) {
      char c;
      fs.read((char*)&c, sizeof(char));
      vocab.dict.at(i).push_back(c);
    }
    vocab.dict.at(i).push_back('\0');
  }
}

} // namespace swan
