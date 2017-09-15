class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        pDict = {}
        for i in p:
            if i in pDict:
                pDict[i]+=1
            else:
                pDict[i]=1
        tempDict = {}
        result = []
        for i in range(0, len(s)-len(p)+1):
            tempDict = {}
            for j in range(i,i+len(p)):
                if s[j] in tempDict:
                    tempDict[s[j]]+=1
                else:
                    tempDict[s[j]]=1
            if tempDict.keys() == pDict.keys():
                flag = True
                for k in tempDict.keys():
                    if tempDict[k] != pDict[k]:
                        flag = False
                        break
                if flag:
                    result.append(i)
        return result


S = Solution()
print S.findAnagrams("ababababab","aab")