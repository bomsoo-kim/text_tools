
#################################################################################################
### Longest Common Tokens: Update 2025-02-04 ####################################################
#################################################################################################
def is_word_char(c, word_char_set=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')):
    return c in word_char_set

if __name__=='__main__':
    print(is_word_char('c'))
    print(is_word_char('-'))
    print(is_word_char('#'))

def find_tokens(text):
    #--- forward cumulutive sum of characters -----------------------
    csum = [0] * len(text) # initialize to zeros
    for i in range(len(text)-1, -1, -1):
        if is_word_char(text[i]):
            if i == len(text)-1: # if the very last charcter
                csum[i] = 1
            else:
                csum[i] = csum[i+1] + 1

    #--- token connectivity = root ----------------------------------
    root = [None] * len(text)
    ii, cnt = {}, 0
    for i in range(len(text)):
        if is_word_char(text[i]):
            if i == 0 or not is_word_char(text[i-1]): # if the initial character of each token
                i_initial = i
                ii[i], cnt = cnt, cnt + 1
            root[i] = i_initial

    return csum, root, ii

if __name__=="__main__":
    print(find_tokens('hello world!'))
    # ([5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0], [0, 0, 0, 0, 0, None, 6, 6, 6, 6, 6, None], {0: 0, 6: 1})

def count_consecutive_initials(ii, i_not_set):
    conn = [0] * len(ii)
    for k,v in ii.items():
        if k not in i_not_set:
            conn[v] = 1

    cnt = 0
    for i in range(len(conn)):
        if conn[i] > 0 and (i == 0 or conn[i-1] == 0):
            cnt += 1

    return cnt


import re

# def compare_text(text1, text2, version='latest', verbose=False): # update: 2024-01-25
def compare_text(text1, text2, version='latest', verbose=False): # update: 2025-02-04
    if version == '1':
        return text1 in text2
    elif version == '2':
        return re.search('(?:^|[^\w])%s(?:$|[^\w])'%re.escape(text1.strip()), text2.strip()) is not None

    csum1, root1, ii1 = find_tokens(text1)
    csum2, root2, ii2 = find_tokens(text2)
    if verbose: 
        print('text1 =', text1)
        print(f'csum1 = {csum1}')
        print(f'root1 = {root1}')
        print(f'ii1 = {ii1}')
        print('\ntext2 =', text2)
        print(f'csum2 = {csum2}')
        print(f'root2 = {root2}')
        print(f'ii2 = {ii2}')

    #--- modified (token-by-token) longest common subsequence algorithm ----------------------------------------------
    dp = [[0]*(len(text2)+1) for _ in range(len(text1)+1)] # space complexity: O(len(text1) * len(text2))
    graph = {} # pointer to the next matched pair or a pathway to it
    max_ij = {} # a set of (i,j) coordinate(s) with the highest dp value, for each token-token pair
    for i in range(len(text1)-1, -1, -1): # time complexity: O(len(text1) * len(text2))
        for j in range(len(text2)-1, -1, -1):
            if not is_word_char(text1[i]) or not is_word_char(text2[j]):
                dp[i][j] = 0 # reset outside each token

            elif text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i+1][j+1] # (token-by-token) longest common subsequence

                graph[(i,j)] = (i+1, j+1)

                k = (root1[i], root2[j])

                if k not in max_ij:
                    max_ij[k] = [] # create if there is no key
                while max_ij[k]: # pop the last element until its dp value is not smaller than that of the current dp
                    a, b = max_ij[k][-1]
                    if dp[a][b] < dp[i][j]: 
                        max_ij[k].pop()
                    else:
                        break
                # max_ij[k].append((i, j)) # BUG! DON'T USE THIS LINE
                if not max_ij[k] or dp[max_ij[k][-1][0]][max_ij[k][-1][1]] <= dp[i][j]:
                    max_ij[k].append((i, j))

            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1]) # (token-by-token) longest common subsequence

                graph[(i,j)] = (i+1, j) if dp[i+1][j] >= dp[i][j+1] else (i, j+1) # if dp[i+1][j] == dp[i][j+1], then select (i+1, j), i.e. downward pointer

    if verbose:
        print('\ndp = ')
        print(f"  [{', '.join(list(text2))}]")
        for c, d in zip(text1, dp): 
            print(c, d)

    if verbose:
        print('\nmax_ij = ')
        for k,v in max_ij.items(): 
            print(f'{k}: {v}')
        print(f'\ngraph = {graph}')


    #--- select the very first match among one or many in each token-token pair [APPROXIMATION] ---------------------------------
    m_ij = {}
    for (ri,rj), vv in max_ij.items(): # (ri,rj) = root (i,j) 
        seen = set()
        for k, (x, y) in enumerate(reversed(vv)):
            match_only = []
            while ((x,y) not in seen) and (dp[x][y] > 0):
                if text1[x] == text2[y]:
                    match_only.append((x,y))
                seen.add((x,y))
                x,y = graph[(x,y)] # next coordinate

            a, b = match_only[-1]
            if dp[a][b] == 1:
                m_ij[(ri,rj,k)] = match_only

    if verbose:
        print('\nm_ij = ')
        for k,v in m_ij.items(): 
            print(f'{k}: {v}')


    #--- prioritize all possible token matches by sorting with certain criteria (greedy method, which may not always give the best) [APPROXIMATION] ---------------
    mm = []
    for (ri, rj, k), vv in m_ij.items():
        i, j = vv[0]
        assert dp[i][j] == len(vv)
        # mm.append((
        #     ri, rj, k,
        #     1*(text1[ri]==text2[rj]), # first character match
        #     dp[i][j], # number of character matches
        #     max(dp[i][j]/csum1[ri], dp[i][j]/csum2[rj]), # max ratio of matched characters in each token
        #     ))
        mm.append({
            'ri': ri, 
            'rj': rj, 
            'k': k,
            # 'is_first_match': 1*(text1[ri]==text2[rj]), # first character match
            'is_first_match': (ri==i) + (rj==j), # first character match for each token pair
            'n_matched': dp[i][j], # number of character matches
            'max_ratio': max(dp[i][j]/csum1[ri], dp[i][j]/csum2[rj]), # max ratio of matched characters in each token
            })
    # mm.sort(key = lambda x: (-x[3], -x[4], -x[5], x[0], x[1], x[2])) # sorting criteria
    mm.sort(key = lambda x: (-x['is_first_match'], -x['n_matched'], -x['max_ratio'], x['ri'], x['rj'], x['k'])) # sorting criteria

    if verbose:
        print('\nmm = ')
        for m in mm:
            print(m)
        print()

    #--- assign the same group number for matched characters --------------------------------------
    n_grp = 0 # group number
    match1, chunk1 = [-1] * len(text1), [-1] * len(text1) # initialize all as unassigned (-1)
    match2, chunk2 = [-1] * len(text2), [-1] * len(text2) # initialize all as unassigned (-1)
    # for ri, rj, k, is_first_match, n_matched, max_ratio in mm: # try matching in this sorted order
    for i_m, row in enumerate(mm): # try matching in this sorted order
        ri, rj, k, is_first_match, n_matched, max_ratio = row['ri'], row['rj'], row['k'], row['is_first_match'], row['n_matched'], row['max_ratio']

        if ((max_ratio == 1) # only if one token is inclusive of or equal to the other: i.e. max_ratio == 1
            # and all(match1[i] < 0 and match2[j] < 0 for i, j in m_ij[(ri,rj,k)]) # if all the detected characters are still unassigned
            and all(chunk1[i] < 0 and chunk2[j] < 0 for i, j in m_ij[(ri,rj,k)]) # if all the detected characters are still unassigned
            ):
            for nn, (i, j) in enumerate(m_ij[(ri,rj,k)]):
                match1[i], match2[j] = n_grp, n_grp # assign the group number for the matched pairs

                #--- find a countinuous chunk of matched characters, to prevent another matching between matched characters: e.g. AA___A__A B CCC (okay), AA_B_A__A _ CCC (not okay)
                if nn == 0:
                    i_prev, j_prev = i,  j
                for ix in range(i_prev, i+1):
                    chunk1[ix] = n_grp
                for jx in range(j_prev, j+1):
                    chunk2[jx] = n_grp
                i_prev, j_prev = i,  j

            mm[i_m]['group_num_assigned'] = n_grp # output for debuging purpose 
            
            n_grp += 1 # change the group number for the next matched pair

    #-------------------------------------------------------------------------
    ii_remaining_1 = set(ii1.keys()).intersection([i for i, (t,m) in enumerate(zip(csum1, match1)) if t > 0 and m < 0]) # if token and not assigned
    ii_remaining_2 = set(ii2.keys()).intersection([i for i, (t,m) in enumerate(zip(csum2, match2)) if t > 0 and m < 0]) # if token and not assigned

    if verbose: print('\nmatch1 =', match1)
    if verbose: print('match2 =', match2)
    if verbose: print('\nii_remaining_1 =', ii_remaining_1, '; ii_remaining_2 =', ii_remaining_2)

    #--- final decision ---------------------------------------------------
    if version == '3':
        is_same = len(ii_remaining_1) == 0 and len(ii_remaining_2) == 0
    else:
        cnt1 = count_consecutive_initials(ii1, ii_remaining_1)
        cnt2 = count_consecutive_initials(ii2, ii_remaining_2)
        if verbose: print('\ncnt1 =', cnt1)
        if verbose: print('cnt2 =', cnt2)
        is_same = (
            (len(ii_remaining_1) == 0 and (len(ii1) <= len(ii2) - len(ii_remaining_2))) or
            (len(ii_remaining_2) == 0 and (len(ii2) <= len(ii1) - len(ii_remaining_1)))
            ) and (cnt1 == 1 and cnt2 == 1)

    if verbose:
        match_out1 = f"{text1}\n{''.join(['^' if n >=0 else ' ' for n in match1])}\n{''.join([chr(ord('A')+n) if n >=0 else ' ' for n in match1])}"
        match_out2 = f"{text2}\n{''.join(['^' if n >=0 else ' ' for n in match2])}\n{''.join([chr(ord('A')+n) if n >=0 else ' ' for n in match2])}"
        print(match_out1)
        print(match_out2)
        print(match_out2)

    return is_same, dp, mm, ii_remaining_1, ii_remaining_2, match1, match2, csum1, csum2, root1, root2

if __name__=='__main__':
    import re
    import pandas as pd

    data = []
    for text1, text2, true_answer in [
        ('International Inn Corporation. Ltd', 'I. I. Co. Limited', True),
        ('International I. Corporation. Ltd', 'I. Inn Co. Limited', True),
        # ('abc', 'abcabc', True), # wierd case !!!!!!
        # ('abc abc', 'abcabc', True), # wierd case !!!!!!
        # ('abc', 'aabbcc', True), # wierd case !!!!!!
        # ('abc abc', 'aabbcc', True), # wierd case !!!!!!
        # ('abc', 'aaaaaabc', True), # wierd case !!!!!!
        # ('xab', 'axb', True), # wierd case !!!!!!
        # ('axbc', 'abcx', True), # wierd case !!!!!!
        # ('abc abc', 'abcabcxa', True), # wierd case !!!!!!
        # ('aecxef ghi', 'aec xef ghi', True), # wierd case !!!!!!
        ]:
        ver1 =  compare_text(text1.lower(), text2.lower(), version='1')
        ver2 =  compare_text(text1.lower(), text2.lower(), version='2')
        ver3, dp, mm, ii_remaining_1, ii_remaining_2, match1, match2, csum1, csum2, root1, root2 = compare_text(text1.lower(), text2.lower(), version='3')
        latest, dp, mm, ii_remaining_1, ii_remaining_2, match1, match2, csum1, csum2, root1, root2 = compare_text(text1.lower(), text2.lower(), verbose=False)
        # latest, dp, mm, ii_remaining_1, ii_remaining_2, match1, match2, csum1, csum2, root1, root2 = compare_text(text1.lower(), text2.lower(), verbose=True)

        match_out1 = f"{text1}\n{''.join(['^' if n >=0 else ' ' for n in match1])}\n{''.join([chr(ord('A')+n) if n >=0 else ' ' for n in match1])}"
        match_out2 = f"{text2}\n{''.join(['^' if n >=0 else ' ' for n in match2])}\n{''.join([chr(ord('A')+n) if n >=0 else ' ' for n in match2])}"

        data.append({'text1':text1, 'text2':text2, 'true_answer':true_answer, 'ver1':ver1, 'ver2':ver2, 'ver3':ver3, 'ver3.1':latest, 'ver2 or ver3':ver2 or ver3, 'ver2 or ver3.1':ver2 or latest, 'match_out':f'{match_out1}\n{match_out2}'})

        print(match_out1)
        print(match_out2)
        print()

    df = pd.DataFrame(data)
    df.to_excel('name_matching.xlsx')
    df

    # with open('escalated_alert_detection_log.txt', 'a') as f:
    #     f.write(text1 + '\n')
    #     f.write(s1 + '\n')
    #     f.write(text2 + '\n')
    #     f.write(s2 + '\n')