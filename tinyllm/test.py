word_decoded = {"hello":[1,2,3,4,5,6,7]}
word_decoded = {"hello":[3,4,3,3,3,3,3,3,3]}
#word_decoded = {"hello":[3,4]}
print(word_decoded)
new_idx = 77
pair_to_merge=(3,4)
token = "hello"
def _merge_pair_and_get_old_adjacent_pairs(new_idx: int, pair_to_merge: Tuple[int, int], token: bytes) -> List[Tuple[int, int]]:
    """Search and find the adjacent pairs in word_decoded table while merge the largest pair that's newly been found.
    """
    old_adjacent_pairs = []
    new_adjacent_pairs = []
    new_id_list = []
    id_list = word_decoded[token]
    id_list_len = len(id_list)
    i: int
    matched = 0

    if id_list_len < 2: raise ValueError("长度小于2（0或1）的token ID序列不可能存在需要合并的字节")
    if id_list_len == 2:
        if pair_to_merge == tuple(id_list):
            matched += 1
            new_id_list = [new_idx]
    else:
        if tuple(id_list[:2]) == pair_to_merge: 
            matched += 1
            new_id_list.append(new_idx)
            if tuple(id_list[2:4]) != pair_to_merge:
                new_adjacent_pairs.append((new_idx, id_list[2]))
                old_adjacent_pairs.append(tuple(id_list[1:3]))
            i = 2
        else:
            new_id_list.append(id_list[0])
            i = 1

        while i < id_list_len-2:
            if tuple(id_list[i:i+2]) == pair_to_merge:
                matched += 1
                new_id_list.append(new_idx)
                old_adjacent_pairs.append(tuple(id_list[i-1:i+1])) # Add the adjacent pairs of two
                new_adjacent_pairs.append((new_id_list[-2], new_idx))
                if tuple(id_list[i+2:i+4]) != pair_to_merge:
                    new_adjacent_pairs.append((new_idx, id_list[i+2]))
                    old_adjacent_pairs.append(tuple(id_list[i+1:i+3]))
                i += 2
            else:
                new_id_list.append(id_list[i])
                i += 1
            
        if i == id_list_len-1: # 说明最后一组被归并了，且此时右边的相邻的pair已经被添加
            new_id_list.append(id_list[-1])
        else: # 否则最后一组没有归并，此时需要将最后一组进行检测
            if tuple(id_list[-2:]) == pair_to_merge: # 最后一组可以归并
                matched += 1
                new_id_list.append(new_idx)
                old_adjacent_pairs.append(tuple(id_list[-3:-1]))
                new_adjacent_pairs.append((new_id_list[-2], new_idx))
            else:
                new_id_list.append(id_list[-2])
                new_id_list.append(id_list[-1])

    return old_adjacent_pairs, new_adjacent_pairs, new_id_list, matched
if __name__ == '__main__':
    print(_merge_pair_and_get_old_adjacent_pairs(new_idx, pair_to_merge, token))
    a, b, c, d = _merge_pair_and_get_old_adjacent_pairs(new_idx, pair_to_merge, token)
    print(a, b, c, d)
