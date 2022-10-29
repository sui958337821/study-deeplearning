
def get_longest_palindromic_substring(text):
    if len(text) <= 1:
        print(1, text)
        return text

    max_len = 0
    substring = ''
    for flag in range(0, len(text)):
        # flag位置作为中间位置
        for distance in range(1, flag + 1):
            if (flag - distance) < 0 or (flag + distance) >= len(text):
                break
            
            if text[flag - distance] == text[flag + distance]:
                if distance*2 + 1 > max_len:
                    substring = text[flag-distance:flag+distance+1]
                    max_len = len(substring)
            else:
                break   
            
        # flag位置作为中间偏左侧位置
        for distance in range(0, flag + 1):
            if (flag - distance) < 0 or (flag + distance + 1) >= len(text):
                break
            
            if text[flag - distance] == text[flag + distance + 1]:
                if distance*2+2 > max_len:
                    substring = text[flag-distance:flag+distance+2]
                    max_len = len(substring)
            else:
                break

    if max_len == 0:
        max_len = 1
        substring = text[0]

    print(max_len, substring)

if __name__ == '__main__':
    get_longest_palindromic_substring('aa')