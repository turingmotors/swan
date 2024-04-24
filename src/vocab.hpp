#ifndef VOCAB_HPP_
#define VOCAB_HPP_

#include <fstream>
#include <string>
#include <vector>

namespace swan {

struct Vocab {
  std::vector<std::string> dict;
};

void ResizeVocab(Vocab& vocab, int vocab_size);
void LoadVocab(Vocab& vocab, std::ifstream& fs);

} // namespace swan

#endif // VOCAB_HPP_
