// Exact targeted colored-state shortcut search with a canonical radius-5 +
// radius-5 meet in the middle.  Unlike the whole-row radius-9 scanner, this
// searches only explicitly ranked windows, keeping memory below 1 GiB.

#define main colored_mitm8_original_main
#include "/private/tmp/colored_mitm8_changed.cpp"
#undef main

namespace {

constexpr uint64_t BLOOM10_BITS = 1ULL << 28;  // 32 MiB.

struct WindowTask {
    int row = -1;
    int start = -1;
    int end = -1;
    double priority = 0.0;
};

struct CompactBack {
    uint64_t hash;
    uint32_t word_index;
};

uint64_t windows_searched10 = 0;
uint64_t forward_states10 = 0;
uint64_t backward_states10 = 0;
uint64_t bloom_passes10 = 0;
uint64_t hash_records10 = 0;
uint64_t exact_hits10 = 0;

void bloom10_add(std::vector<uint64_t>& bloom, uint64_t hash) {
    const uint64_t first = hash & (BLOOM10_BITS - 1);
    const uint64_t second = mix64(hash ^ 0x9e3779b97f4a7c15ULL) & (BLOOM10_BITS - 1);
    bloom[first >> 6] |= 1ULL << (first & 63);
    bloom[second >> 6] |= 1ULL << (second & 63);
}

bool bloom10_maybe(const std::vector<uint64_t>& bloom, uint64_t hash) {
    const uint64_t first = hash & (BLOOM10_BITS - 1);
    const uint64_t second = mix64(hash ^ 0x9e3779b97f4a7c15ULL) & (BLOOM10_BITS - 1);
    return ((bloom[first >> 6] >> (first & 63)) & 1U) &&
           ((bloom[second >> 6] >> (second & 63)) & 1U);
}

std::vector<WindowTask> load_window_plan(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open window plan " + path);
    std::vector<WindowTask> tasks;
    std::string line;
    std::getline(input, line);  // Header: row, start, end, priority.
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream fields(line);
        WindowTask task;
        if (!(fields >> task.row >> task.start >> task.end >> task.priority)) {
            throw std::runtime_error("bad window plan row: " + line);
        }
        tasks.push_back(task);
    }
    return tasks;
}

Bridge search_window10(
    const WindowTask& task,
    const std::vector<State>& positions,
    const std::vector<Word>& words5) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<CompactBack> backward;
    backward.reserve(words5.size());
    std::vector<uint64_t> bloom(BLOOM10_BITS / 64);
    State state;

    for (uint32_t index = 0; index < words5.size(); ++index) {
        const uint64_t hash = apply_perm_hash(positions[task.end], words5[index].perm, state);
        backward.push_back(CompactBack{hash, index});
        bloom10_add(bloom, hash);
        ++backward_states10;
    }
    std::sort(backward.begin(), backward.end(), [](const CompactBack& left, const CompactBack& right) {
        if (left.hash != right.hash) return left.hash < right.hash;
        return left.word_index < right.word_index;
    });

    Bridge best;
    for (const Word& forward : words5) {
        const uint64_t hash = apply_perm_hash(positions[task.start], forward.perm, state);
        ++forward_states10;
        if (!bloom10_maybe(bloom, hash)) continue;
        ++bloom_passes10;
        auto iterator = std::lower_bound(
            backward.begin(), backward.end(), hash,
            [](const CompactBack& record, uint64_t value) { return record.hash < value; });
        for (; iterator != backward.end() && iterator->hash == hash; ++iterator) {
            ++hash_records10;
            const Word& reverse = words5[iterator->word_index];
            State reverse_state;
            apply_perm_hash(positions[task.end], reverse.perm, reverse_state);
            if (reverse_state != state) continue;
            ++exact_hits10;
            std::vector<uint8_t> bridge = construct_bridge(forward, reverse);
            if (int(bridge.size()) < best.len && int(bridge.size()) < task.end - task.start) {
                best.len = int(bridge.size());
                best.moves = std::move(bridge);
            }
        }
    }
    ++windows_searched10;
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "row=" << task.row << " start=" << task.start << " end=" << task.end
              << " old_len=" << task.end - task.start
              << " best_len=" << (best.len < 100 ? best.len : -1)
              << " priority=" << task.priority << " seconds=" << seconds << std::endl;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: colored_window_mitm10 puzzle_info candidate test plan.tsv out.csv relations.tsv\n";
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    load_puzzle_info(argv[1]);
    load_rows(argv[2]);
    load_test(argv[3]);
    if (rows.size() != initial_states.size()) throw std::runtime_error("row count mismatch");
    const std::vector<WindowTask> tasks = load_window_plan(argv[4]);
    if (tasks.empty()) throw std::runtime_error("empty window plan");

    std::vector<Word> words5 = generate_words(5);
    if (words5.size() != 3512239) throw std::runtime_error("radius-5 word count mismatch");
    std::cout << "tasks=" << tasks.size() << " words5=" << words5.size()
              << " sizeof_word=" << sizeof(Word)
              << " sizeof_back=" << sizeof(CompactBack) << std::endl;

    std::vector<std::vector<std::vector<Bridge>>> best(rows.size());
    int active_row = -1;
    std::vector<State> positions;
    for (const WindowTask& task : tasks) {
        if (task.row < 0 || task.row >= int(rows.size())) throw std::runtime_error("bad planned row");
        const int length = int(rows[task.row].moves.size());
        if (task.start < 0 || task.end > length || task.start >= task.end) {
            throw std::runtime_error("bad planned window bounds");
        }
        if (task.end - task.start < 11) throw std::runtime_error("planned window cannot improve at radius 10");
        if (best[task.row].empty()) {
            best[task.row].assign(length + 1, std::vector<Bridge>(length + 1));
        }
        if (active_row != task.row) {
            active_row = task.row;
            positions = path_states_for(task.row);
        }
        Bridge candidate = search_window10(task, positions, words5);
        Bridge& destination = best[task.row][task.start][task.end];
        if (candidate.len < destination.len) destination = std::move(candidate);
    }

    std::ofstream output(argv[5]);
    std::ofstream relations(argv[6]);
    if (!output || !relations) throw std::runtime_error("cannot create output files");
    output << "initial_state_id,path\n";
    relations << "row\tstart\tend\told_len\tnew_len\tgain\told_path\tnew_path\n";

    long long before_total = 0;
    long long after_total = 0;
    int improved_rows = 0;
    int selected_relations = 0;
    for (const Row& row : rows) {
        const int length = int(row.moves.size());
        std::vector<uint8_t> result;
        if (best[row.id].empty()) {
            result = row.moves;
        } else {
            std::vector<int> gain(length + 1, 0);
            std::vector<int> chosen_end(length, -1);
            for (int start = length - 1; start >= 0; --start) {
                gain[start] = gain[start + 1];
                for (int end = start + 1; end <= length; ++end) {
                    const Bridge& bridge = best[row.id][start][end];
                    if (bridge.len >= 100) continue;
                    const int candidate_gain = end - start - bridge.len + gain[end];
                    if (candidate_gain > gain[start]) {
                        gain[start] = candidate_gain;
                        chosen_end[start] = end;
                    }
                }
            }
            for (int index = 0; index < length;) {
                const int end = chosen_end[index];
                if (end < 0) {
                    result.push_back(row.moves[index++]);
                    continue;
                }
                const Bridge& bridge = best[row.id][index][end];
                std::vector<uint8_t> old_piece(row.moves.begin() + index, row.moves.begin() + end);
                relations << row.id << '\t' << index << '\t' << end << '\t' << end - index
                          << '\t' << bridge.len << '\t' << end - index - bridge.len << '\t'
                          << format_path(old_piece) << '\t' << format_path(bridge.moves) << '\n';
                result.insert(result.end(), bridge.moves.begin(), bridge.moves.end());
                ++selected_relations;
                index = end;
            }
            result = normalize_same_axis(result);
        }

        State final_state = initial_states[row.id];
        for (uint8_t move : result) final_state = apply_move(final_state, move);
        if (final_state != central_state) {
            throw std::runtime_error("rewritten path unsolved at row " + std::to_string(row.id));
        }
        before_total += row.moves.size();
        after_total += result.size();
        if (result.size() < row.moves.size()) ++improved_rows;
        output << row.id << ',' << format_path(result) << '\n';
    }

    std::cout << "windows_searched=" << windows_searched10 << '\n'
              << "forward_states=" << forward_states10 << '\n'
              << "backward_states=" << backward_states10 << '\n'
              << "bloom_passes=" << bloom_passes10 << '\n'
              << "hash_records=" << hash_records10 << '\n'
              << "exact_hits=" << exact_hits10 << '\n'
              << "selected_relations=" << selected_relations << '\n'
              << "improved_rows=" << improved_rows << '\n'
              << "before_total=" << before_total << '\n'
              << "after_total=" << after_total << '\n'
              << "total_gain=" << before_total - after_total << '\n'
              << "total_seconds=" << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - started).count() << '\n';
    return 0;
}
