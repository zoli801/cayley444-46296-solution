#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using Perm = std::array<uint8_t, 96>;
using State = std::array<uint8_t, 96>;

struct Row {
    int id = -1;
    std::vector<uint8_t> moves;
};

struct Word {
    Perm perm{};
    uint32_t packed = 0;
    uint8_t len = 0;
};

struct BackRecord {
    uint64_t hash = 0;
    uint32_t word_index = 0;
    uint8_t end = 0;
};

struct Bridge {
    int len = 100;
    std::vector<uint8_t> moves;
};

static std::array<Perm, 24> generators;
static std::array<std::array<uint64_t, 6>, 96> zobrist;
static std::vector<Row> rows;
static std::vector<State> initial_states;
static State central_state;

static uint64_t forward_states = 0;
static uint64_t backward_states = 0;
static uint64_t bloom_passes = 0;
static uint64_t records_checked = 0;
static uint64_t exact_state_hits = 0;

constexpr uint64_t BLOOM_BITS = 1ULL << 23; // 1 MiB per row.

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static int move_id(const std::string& token) {
    if (token.empty()) throw std::runtime_error("empty move token");
    int neg = token[0] == '-' ? 1 : 0;
    if (int(token.size()) != neg + 2) throw std::runtime_error("bad move " + token);
    int axis = token[neg] == 'f' ? 0 : token[neg] == 'r' ? 1 : token[neg] == 'd' ? 2 : -1;
    int layer = token[neg + 1] - '0';
    if (axis < 0 || layer < 0 || layer > 3) throw std::runtime_error("bad move " + token);
    return axis * 8 + layer * 2 + neg;
}

static std::string move_name(int id) {
    static constexpr char axes[] = {'f', 'r', 'd'};
    std::string result;
    if (id & 1) result.push_back('-');
    result.push_back(axes[id / 8]);
    result.push_back(char('0' + (id % 8) / 2));
    return result;
}

static std::vector<uint8_t> parse_path(const std::string& text) {
    std::vector<uint8_t> result;
    if (text.empty()) return result;
    size_t begin = 0;
    while (true) {
        size_t end = text.find('.', begin);
        result.push_back(uint8_t(move_id(text.substr(begin, end == std::string::npos ? end : end - begin))));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

static std::string format_path(const std::vector<uint8_t>& path) {
    std::ostringstream out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) out << '.';
        out << move_name(path[i]);
    }
    return out.str();
}

static void parse_number_array(const std::string& text, size_t pos, uint8_t* output) {
    for (int i = 0; i < 96; ++i) {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n')) ++pos;
        int value = 0;
        bool saw = false;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            saw = true;
            value = value * 10 + text[pos++] - '0';
        }
        if (!saw || value < 0 || value >= 96) throw std::runtime_error("bad numeric array");
        output[i] = uint8_t(value);
        if (i != 95) {
            if (pos >= text.size() || text[pos] != ',') throw std::runtime_error("bad numeric array separator");
            ++pos;
        }
    }
}

static void load_puzzle_info(const std::string& path) {
    std::string text = slurp(path);
    std::string central_needle = "\"central_state\":[";
    size_t cp = text.find(central_needle);
    if (cp == std::string::npos) throw std::runtime_error("missing central state");
    parse_number_array(text, cp + central_needle.size(), central_state.data());
    for (int id = 0; id < 24; ++id) {
        std::string needle = "\"" + move_name(id) + "\":[";
        size_t pos = text.find(needle);
        if (pos == std::string::npos) throw std::runtime_error("missing generator " + move_name(id));
        parse_number_array(text, pos + needle.size(), generators[id].data());
    }
    uint64_t seed = 0x6a09e667f3bcc909ULL;
    for (int i = 0; i < 96; ++i) {
        for (int c = 0; c < 6; ++c) {
            seed += 0x9e3779b97f4a7c15ULL;
            zobrist[i][c] = mix64(seed);
        }
    }
}

static void load_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t comma = line.find(',');
        int id = std::stoi(line.substr(0, comma));
        if (id != int(rows.size())) throw std::runtime_error("noncontiguous candidate IDs");
        rows.push_back(Row{id, parse_path(line.substr(comma + 1))});
    }
}

static State parse_color_state(const std::string& text) {
    State result{};
    size_t pos = 0;
    for (int i = 0; i < 96; ++i) {
        int value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            value = value * 10 + text[pos++] - '0';
        }
        if (value < 0 || value >= 6) throw std::runtime_error("bad color");
        result[i] = uint8_t(value);
        if (i != 95) {
            if (pos >= text.size() || text[pos] != ',') throw std::runtime_error("bad test state");
            ++pos;
        }
    }
    return result;
}

static void load_test(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t comma = line.find(',');
        int id = std::stoi(line.substr(0, comma));
        size_t q1 = line.find('"', comma + 1);
        size_t q2 = line.find('"', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) throw std::runtime_error("bad test CSV");
        if (id != int(initial_states.size())) throw std::runtime_error("noncontiguous test IDs");
        initial_states.push_back(parse_color_state(line.substr(q1 + 1, q2 - q1 - 1)));
    }
}

static Perm identity_perm() {
    Perm p{};
    for (int i = 0; i < 96; ++i) p[i] = uint8_t(i);
    return p;
}

static Perm append_move(const Perm& current, int move) {
    Perm out;
    const Perm& g = generators[move];
    for (int i = 0; i < 96; ++i) out[i] = current[g[i]];
    return out;
}

static State apply_move(const State& state, int move) {
    State out;
    const Perm& g = generators[move];
    for (int i = 0; i < 96; ++i) out[i] = state[g[i]];
    return out;
}

static uint64_t apply_perm_hash(const State& state, const Perm& p, State& out) {
    uint64_t h = 0;
    for (int i = 0; i < 96; ++i) {
        out[i] = state[p[i]];
        h ^= zobrist[i][out[i]];
    }
    return h;
}

static std::vector<uint8_t> unpack_word(const Word& word) {
    std::vector<uint8_t> out;
    for (int i = 0; i < word.len; ++i) out.push_back(uint8_t((word.packed >> (5 * i)) & 31));
    return out;
}

static void generate_words_dfs(std::vector<Word>& out, int max_depth, const Perm& current,
                               int depth, int last_axis, int last_layer,
                               int current_layer_count, int current_layer_sign,
                               uint32_t packed) {
    out.push_back(Word{current, packed, uint8_t(depth)});
    if (depth == max_depth) return;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != last_axis) {
            for (int layer = 0; layer < 4; ++layer) {
                for (int sign = 0; sign < 2; ++sign) {
                    int id = axis * 8 + layer * 2 + sign;
                    Perm child = append_move(current, id);
                    generate_words_dfs(out, max_depth, child, depth + 1, axis, layer, 1, sign,
                                       packed | (uint32_t(id) << (5 * depth)));
                }
            }
        } else {
            if (current_layer_sign == 0 && current_layer_count == 1) {
                int id = axis * 8 + last_layer * 2;
                Perm child = append_move(current, id);
                generate_words_dfs(out, max_depth, child, depth + 1, axis, last_layer, 2, 0,
                                   packed | (uint32_t(id) << (5 * depth)));
            }
            for (int layer = last_layer + 1; layer < 4; ++layer) {
                for (int sign = 0; sign < 2; ++sign) {
                    int id = axis * 8 + layer * 2 + sign;
                    Perm child = append_move(current, id);
                    generate_words_dfs(out, max_depth, child, depth + 1, axis, layer, 1, sign,
                                       packed | (uint32_t(id) << (5 * depth)));
                }
            }
        }
    }
}

static std::vector<Word> generate_words(int max_depth) {
    std::vector<Word> result;
    generate_words_dfs(result, max_depth, identity_perm(), 0, -1, -1, 0, 0, 0);
    return result;
}

static std::vector<uint8_t> normalize_same_axis(const std::vector<uint8_t>& path) {
    std::vector<uint8_t> result;
    size_t i = 0;
    while (i < path.size()) {
        int axis = path[i] / 8;
        int exp[4] = {0, 0, 0, 0};
        size_t j = i;
        while (j < path.size() && path[j] / 8 == axis) {
            int layer = (path[j] % 8) / 2;
            exp[layer] = (exp[layer] + ((path[j] & 1) ? 3 : 1)) & 3;
            ++j;
        }
        for (int layer = 0; layer < 4; ++layer) {
            int positive = axis * 8 + layer * 2;
            if (exp[layer] == 1) result.push_back(uint8_t(positive));
            else if (exp[layer] == 2) {
                result.push_back(uint8_t(positive));
                result.push_back(uint8_t(positive));
            } else if (exp[layer] == 3) result.push_back(uint8_t(positive + 1));
        }
        i = j;
    }
    return result;
}

static std::vector<uint8_t> construct_bridge(const Word& forward, const Word& backward) {
    std::vector<uint8_t> result = unpack_word(forward);
    std::vector<uint8_t> back = unpack_word(backward);
    for (auto it = back.rbegin(); it != back.rend(); ++it) result.push_back(uint8_t((*it) ^ 1));
    return normalize_same_axis(result);
}

static void bloom_add(std::vector<uint64_t>& bloom, uint64_t h) {
    uint64_t a = h & (BLOOM_BITS - 1);
    uint64_t b = mix64(h ^ 0x9e3779b97f4a7c15ULL) & (BLOOM_BITS - 1);
    bloom[a >> 6] |= 1ULL << (a & 63);
    bloom[b >> 6] |= 1ULL << (b & 63);
}

static bool bloom_maybe(const std::vector<uint64_t>& bloom, uint64_t h) {
    uint64_t a = h & (BLOOM_BITS - 1);
    uint64_t b = mix64(h ^ 0x9e3779b97f4a7c15ULL) & (BLOOM_BITS - 1);
    return ((bloom[a >> 6] >> (a & 63)) & 1) && ((bloom[b >> 6] >> (b & 63)) & 1);
}

static std::vector<State> path_states_for(int row_id) {
    const auto& path = rows[row_id].moves;
    std::vector<State> states(path.size() + 1);
    states[0] = initial_states[row_id];
    for (size_t i = 0; i < path.size(); ++i) states[i + 1] = apply_move(states[i], path[i]);
    if (states.back() != central_state) throw std::runtime_error("input candidate unsolved at row " + std::to_string(row_id));
    return states;
}

static std::vector<std::vector<Bridge>> search_row(int row_id, const std::vector<Word>& words4,
                                                   const std::vector<Word>& reverse_words) {
    auto row_begin = std::chrono::steady_clock::now();
    const int n = int(rows[row_id].moves.size());
    std::vector<State> positions = path_states_for(row_id);
    std::vector<BackRecord> backward;
    backward.reserve((n - 7) * reverse_words.size());
    std::vector<uint64_t> bloom(BLOOM_BITS / 64);

    State state;
    for (int end = 8; end <= n; ++end) {
        for (uint32_t wi = 0; wi < reverse_words.size(); ++wi) {
            uint64_t h = apply_perm_hash(positions[end], reverse_words[wi].perm, state);
            backward.push_back(BackRecord{h, wi, uint8_t(end)});
            bloom_add(bloom, h);
            ++backward_states;
        }
    }
    std::sort(backward.begin(), backward.end(), [](const BackRecord& a, const BackRecord& b) {
        return a.hash < b.hash;
    });

    std::vector<std::vector<Bridge>> best(n + 1, std::vector<Bridge>(n + 1));
    for (int start = 0; start <= n - 8; ++start) {
        for (const Word& fw : words4) {
            uint64_t h = apply_perm_hash(positions[start], fw.perm, state);
            ++forward_states;
            if (!bloom_maybe(bloom, h)) continue;
            ++bloom_passes;
            auto first = std::lower_bound(backward.begin(), backward.end(), h,
                [](const BackRecord& rec, uint64_t value) { return rec.hash < value; });
            for (auto it = first; it != backward.end() && it->hash == h; ++it) {
                ++records_checked;
                const Word& bw = reverse_words[it->word_index];
                if (int(fw.len) + int(bw.len) > 8 || int(it->end) <= start) continue;
                State reverse_state;
                apply_perm_hash(positions[it->end], bw.perm, reverse_state);
                if (reverse_state != state) continue;
                ++exact_state_hits;
                std::vector<uint8_t> bridge = construct_bridge(fw, bw);
                int window_len = int(it->end) - start;
                if (int(bridge.size()) >= window_len) continue;
                Bridge& slot = best[start][it->end];
                if (int(bridge.size()) < slot.len) {
                    slot.len = int(bridge.size());
                    slot.moves = std::move(bridge);
                }
            }
        }
    }
    auto row_end = std::chrono::steady_clock::now();
    int intervals = 0;
    int max_gain = 0;
    for (int i = 0; i <= n; ++i) for (int j = i + 1; j <= n; ++j) {
        if (best[i][j].len < 100) {
            ++intervals;
            max_gain = std::max(max_gain, j - i - best[i][j].len);
        }
    }
    std::cout << "row=" << row_id << " len=" << n << " improving_intervals=" << intervals
              << " max_gain=" << max_gain << " seconds="
              << std::chrono::duration<double>(row_end - row_begin).count() << std::endl;
    return best;
}

int main(int argc, char** argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: colored_mitm7 puzzle_info candidate test out.csv relations.tsv min_path_len max_path_len [comma_ids]\n";
        return 2;
    }
    auto begin = std::chrono::steady_clock::now();
    load_puzzle_info(argv[1]);
    load_rows(argv[2]);
    load_test(argv[3]);
    if (rows.size() != initial_states.size()) throw std::runtime_error("row count mismatch");
    int min_path_len = std::stoi(argv[6]);
    int max_path_len = std::stoi(argv[7]);
    std::unordered_set<int> selected_ids;
    if (argc == 9) {
        std::istringstream ids(argv[8]);
        std::string item;
        while (std::getline(ids, item, ',')) {
            if (!item.empty()) selected_ids.insert(std::stoi(item));
        }
    }

    std::vector<Word> words4 = generate_words(4);
    if (words4.size() != 182599) throw std::runtime_error("word count mismatch");
    std::cout << "words_radius4_forward=" << words4.size()
              << " words_radius4_reverse=" << words4.size() << std::endl;

    std::vector<std::vector<std::vector<Bridge>>> all_bridges(rows.size());
    int searched_rows = 0;
    for (const Row& row : rows) {
        if (int(row.moves.size()) < min_path_len || int(row.moves.size()) > max_path_len) continue;
        if (!selected_ids.empty() && !selected_ids.count(row.id)) continue;
        all_bridges[row.id] = search_row(row.id, words4, words4);
        ++searched_rows;
    }

    std::ofstream out(argv[4]);
    std::ofstream rel(argv[5]);
    if (!out || !rel) throw std::runtime_error("cannot create outputs");
    out << "initial_state_id,path\n";
    rel << "row\tstart\tend\told_len\tnew_len\tgain\told_path\tnew_path\n";
    long long before_total = 0, after_total = 0;
    int improved_rows = 0, selected_relations = 0;

    for (const Row& row : rows) {
        const int n = int(row.moves.size());
        std::vector<uint8_t> result;
        if (all_bridges[row.id].empty()) {
            result = row.moves;
        } else {
            auto& best = all_bridges[row.id];
            std::vector<int> dp(n + 1, 0), chosen_end(n, -1);
            for (int i = n - 1; i >= 0; --i) {
                dp[i] = dp[i + 1];
                for (int end = i + 1; end <= n; ++end) {
                    if (best[i][end].len >= 100) continue;
                    int gain = end - i - best[i][end].len;
                    int value = gain + dp[end];
                    if (value > dp[i]) {
                        dp[i] = value;
                        chosen_end[i] = end;
                    }
                }
            }
            for (int i = 0; i < n;) {
                int end = chosen_end[i];
                if (end < 0) {
                    result.push_back(row.moves[i++]);
                    continue;
                }
                const Bridge& b = best[i][end];
                std::vector<uint8_t> old_piece(row.moves.begin() + i, row.moves.begin() + end);
                rel << row.id << '\t' << i << '\t' << end << '\t' << (end - i) << '\t'
                    << b.len << '\t' << (end - i - b.len) << '\t' << format_path(old_piece)
                    << '\t' << format_path(b.moves) << '\n';
                result.insert(result.end(), b.moves.begin(), b.moves.end());
                ++selected_relations;
                i = end;
            }
            result = normalize_same_axis(result);
        }

        State state = initial_states[row.id];
        for (uint8_t m : result) state = apply_move(state, m);
        if (state != central_state) throw std::runtime_error("rewritten path unsolved at row " + std::to_string(row.id));
        before_total += row.moves.size();
        after_total += result.size();
        if (result.size() < row.moves.size()) ++improved_rows;
        out << row.id << ',' << format_path(result) << '\n';
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "searched_rows=" << searched_rows << '\n';
    std::cout << "forward_states=" << forward_states << '\n';
    std::cout << "backward_states=" << backward_states << '\n';
    std::cout << "bloom_passes=" << bloom_passes << '\n';
    std::cout << "records_checked=" << records_checked << '\n';
    std::cout << "exact_state_hits=" << exact_state_hits << '\n';
    std::cout << "selected_relations=" << selected_relations << '\n';
    std::cout << "improved_rows=" << improved_rows << '\n';
    std::cout << "before_total=" << before_total << '\n';
    std::cout << "after_total=" << after_total << '\n';
    std::cout << "total_gain=" << before_total - after_total << '\n';
    std::cout << "total_seconds=" << std::chrono::duration<double>(end - begin).count() << '\n';
    return 0;
}
