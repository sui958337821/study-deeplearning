class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) + len(s2) != len(s3):
            return False
        if len(s1) == 0 or len(s2) == 0:
            if len(s1) == 0 and s2 == s3:
                return True
            elif s1 == s3 and len(s2) == 0:
                return True
            else:
                return False

        dp = [[0] * (len(s1) + 1)] * (len(s2) + 1)
        dp = [[0] * (len(s1) + 1) for _ in range((len(s2) + 1))]

        dp[0][0] = 1

        # 初始化s2的边
        for i in range(1, len(s2) + 1):
            if dp[i-1][0] == 1 and s2[i-1] == s3[i-1]:
                dp[i][0] = 1
            else:
                break

        for j in range(1, len(s1) + 1):
            if dp[0][j-1] == 1 and s1[j-1] == s3[j-1]:
                dp[0][j] = 1
            else:
                break

        for i in range(1, len(s2)+1):
            for j in range(1, len(s1)+1):
                print(i,j)
                if dp[i][j-1] != 0 or dp[i-1][j] != 0:
                    if dp[i-1][j]!= 0 and s2[i-1] == s3[i+j-1]:
                        dp[i][j] = 1
                        continue
                    elif dp[i][j-1] != 0 and s1[j-1] == s3[i+j-1]:
                        dp[i][j] = 1

        return dp[len(s2)][len(s1)] == 1

a = Solution()
print(a.isInterleave("aabcc", "dbbca", "aadbbcbcac"))