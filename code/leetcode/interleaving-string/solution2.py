# DFS
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False

        def dps(i, j):
            if i == len(s1) and j == len(s2):
                return True

            choose_1 = False
            choose_2 = False

            if i < len(s1) and s1[i] == s3[i+j]:
                choose_1 = dps(i + 1, j)
            if j < len(s2) and s2[j] == s3[i+j]:
                choose_2 = dps(i, j + 1)

            return choose_1 or choose_2

        return dps(0, 0)

a = Solution()
print(a.isInterleave("aabcc", "dbbca", "aadbbcbcac"))