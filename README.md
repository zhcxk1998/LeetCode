# LeetCode
 用于记录LeetCode上的做题记录

[toc]

---

# 3. 无重复字符的最长子串（中等）
给定一个字符串，请你找出其中不含有重复字符的最长子串的长度。

示例1:
```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是"wke"，所以其长度为 3。
    请注意，你的答案必须是 子串 的长度，"pwke"是一个子序列，不是子串。

```
```javascript
/**
 * 方法一，滑动窗口，
 *
 * @param {string} s
 * @return {number}
 */
var lengthOfLongestSubstring = function (s) {
  let [max, start, end, visited] = [0, 0, 0, new Set()]

  while (start < s.length && end < s.length) {
    if (visited.has(s[end])) {
      visited.delete(s[start])
      start += 1
    } else {
      visited.add(s[end])
      end += 1
      max = Math.max(max, end - start)
    }
  }
  return max
};

/**
 * @param {string} s
 * @return {number}
 */
var lengthOfLongestSubstring = function (s) {
  let [max, start, end, visited] = [0, 0, 0, new Map()]
  while (end < s.length) {
    if (visited.has(s[end])) {
      start = Math.max(start, visited.get(s[end]))
    }
    visited.set(s[end], end + 1)
    max = Math.max(max, end - start + 1)
    end += 1
  }
  return max
};

```

---

# 13. 罗马数字转整形（简单）
罗马数字包含以下七种字符:I  ，V，X，L，C，D和M。
```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做II，即为两个并列的 1。12 写做XII，即为X+II。 27 写做XXVII, 即为XX+V+II。
```
通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做IIII，而是IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为IX。这个特殊的规则只适用于以下六种情况：
```
I可以放在V(5) 和X(10) 的左边，来表示 4 和 9。
X可以放在L(50) 和C(100) 的左边，来表示 40 和90。
C可以放在D(500) 和M(1000) 的左边，来表示400 和900。
给定一个罗马数字，将其转换成整数。输入确保在 1到 3999 的范围内。
```
示例1:
```
输入:"III"
输出: 3
```
示例2:
```
输入:"IV"
输出: 4
```
示例3:
```
输入:"IX"
输出: 9
```
示例4:
```
输入:"LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```
示例5:
```
输入:"MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```
```javascript
/**
 * @param {string} s
 * @return {number}
 */
var romanToInt = function (s) {
  const valueMap = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000
  }
  let res = 0
  const len = s.length
  for (let i = 0; i < len - 1; i++) {
    const cur = valueMap[s[i]]
    if (valueMap[s[i]] < valueMap[s[i + 1]]) {
      res -= cur
    } else {
      res += cur
    }
  }
  res += valueMap[s[len - 1]]
  return res
};
```

---


# 14. 最长公共前缀（简单）
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串""。
```
示例1:

输入: ["flower","flow","flight"]
输出: "fl"
示例2:

输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
说明:

所有输入只包含小写字母a-z。
```
```javascript
/**
 * @param {string[]} strs
 * @return {string}
 */
var longestCommonPrefix = function (strs) {
  if (strs.length === 0) return ''
  let res = ""
  let len = strs.length
  strs.sort((a, b) => a.length - b.length)
  for (let i = 0; i < strs[0].length; i++) {
    let pre = strs[0][i]
    let flag = true
    for (let j = 1; j < len; j++) {
      if (strs[j][i] !== pre) {
        flag = false
        break
      }
    }
    if (flag) {
      res += pre
    } else {
      break
    }
  }
  return res
};

```

---

# 19. 删除链表的倒数第N个节点（中等）
给定一个链表，删除链表的倒数第n个节点，并且返回链表的头结点。
```
示例：

给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
说明：

给定的 n保证是有效的。

进阶：

你能尝试使用一趟扫描实现吗？

```
```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} n
 * @return {ListNode}
 */
var removeNthFromEnd = function (head, n) {
  let res = head

  while (res) {
    let count = n
    let temp = res
    while (count !== 0) {
      temp = temp.next
      count -= 1
    }
    if (!temp) {
      head = res.next
      return head
    }
    if (!temp.next) {
      res.next = res.next.next
      return head
    }
    res = res.next
  }

  return head
};

```

---

# 21. 合并两个有序链表（简单）
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
```
示例：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

```
```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var mergeTwoLists = function (l1, l2) {
  let res = new ListNode(0)
  let p = res

  while (l1 && l2) {
    if (l1.val <= l2.val) {
      p.next = l1
      l1 = l1.next
    } else {
      p.next = l2
      l2 = l2.next
    }
    p = p.next
  }
  p.next = l1 || l2

  return res.next
};

```

---


# 22. 括号生成（中等）
给出n代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
```
例如，给出n = 3，生成结果为：

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

```
```javascript
/**
 * 当左括号大于有效括号的时候就可以添加右括号
 *
 * @param {number} n
 * @return {string[]}
 */
var generateParenthesis = function (n) {
  let res = []
  let path = ''   // 生成括号组合
  let isValid = 0 // 有效的括号数量
  let left = 0    // 左括号的数量
  getPath(isValid, left, n, path, res)
  return res
};

var getPath = function (isValid, left, len, path, res) {
  if (isValid === len) {
    res.push(path)
    return
  }

  if (left < len) {
    getPath(isValid, left + 1, len, path + '(', res)
  }

  if (left > isValid) {
    getPath(isValid + 1, left, len, path + ')', res)
  }
}

```

---



# 36. 有效的数独（中等）
判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。

![image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

上图是一个部分填充的有效的数独。

数独部分空格内已填入了数字，空白格用 '.' 表示。
```
示例 1:

输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true
示例 2:

输入:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: false
解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
     但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
说明:

一个有效的数独（部分已被填充）不一定是可解的。
只需要根据以上规则，验证已经填入的数字是否有效即可。
给定数独序列只包含数字 1-9 和字符 '.' 。
给定数独永远是 9x9 形式的。
```
```javascript
/**
 * @param {character[][]} board
 * @return {boolean}
 */
var isValidSudoku = function (board) {
  for (let row = 0; row < board.length; row++) {
    for (let col = 0; col < board[row].length; col++) {
      let cur = board[row][col]
      if (cur === ".") continue
      // 判断行
      if (board[row].indexOf(cur) !== board[row].lastIndexOf(cur)) {
        return false
      }
      
      // 判断列
      let count = 0
      for (let i = 0; i < board.length; i++) {
        if (cur === board[i][col]) {
          count += 1
        }
        if (count >= 2) {
          return false
        }
      }

      // 判断区域
      let start = {
        row: parseInt(row / 3) * 3,
        col: parseInt(col / 3) * 3
      }

      let end = {
        row: start.row + 3,
        col: start.col + 3
      }

      count = 0
      for (let i = start.row; i < end.row; i++) {
        for (let j = start.col; j < end.col; j++) {
          if (board[i][j] === cur) {
            count += 1
          }
          if (count >= 2) {
            return false
          }
        }
      }
    }
  }
  return true
};
```

---


# 39. 组合总和（中等）
给定一个无重复元素的数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。

candidates中的数字可以无限制重复被选取。

说明：
```
所有数字（包括target）都是正整数。
解集不能包含重复的组合。
示例1:

输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
示例2:

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
 [2,2,2,2],
 [2,3,3],
 [3,5]
]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/combination-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
```javascript
/**
 * @param {number[]} candidates
 * @param {number} target
 * @return {number[][]}
 */
var combinationSum = function (candidates, target) {
  let res = []
  let path = []
  let sum = 0
  backTrack(candidates, target, sum, path, res, 0)
  return res
};

var backTrack = function (candidates, target, sum, path, res, start) {
  /* 总和大于目标值，就没有必要再继续了 */
  if (sum > target) {
    return
  }

  /* 总和等于目标值，则添加进数组 */
  if (sum === target) {
    res.push([...path])
    return
  }

  /* 这里let i = start，以及函数传入的start，是为了不重复寻找。期初我let i = 0导致会重复寻找相同的组合，因为元素是不重复的，所以我们让起始的i等于上一次的i，即可避免重复寻找相同的组合 */
  for (let i = start; i < candidates.length; i++) {
    let cur = candidates[i]
    path.push(cur)
    backTrack(candidates, target, sum + cur, path, res, i)
    path.pop()
  }
}


```

# 46. 全排列（中等）
给定一个没有重复数字的序列，返回其所有可能的全排列。
```
示例:

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

```
```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */

var permute = function (nums) {
  let path = []
  let visited = Array.from({ length: nums.length }, _ => false)
  let res = []
  DFS(nums, 0, nums.length, path, visited, res)
  return res
};

var DFS = function (nums, curSize, len, path, visited, res) {
  if (curSize === len) {
    res.push([...path])
    return
  }

  for (let i = 0; i < len; i++) {
    if (!visited[i]) {
      visited[i] = true
      path.push(nums[i])
      DFS(nums, curSize + 1, len, path, visited, res)
      visited[i] = false
      path.pop()
    }
  }
}
```

---


# 48. 旋转图像（中等）
给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

说明：

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
```
示例 1:

给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
示例 2:

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```
```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsets = function (nums) {
  let res = [[]];
  for (let i = 0; i < nums.length; i++) {
    res.map(e=>{
      res.push(e.concat(nums[i]))
    })
  }
  return res;
};
```

---

# 49. 字母异位词分组（中等）
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
```
示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。
```
```javascript
/**
 * @param {string[]} strs
 * @return {string[][]}
 */
var groupAnagrams = function (strs) {
  let res = []
  let visited = new Map()
  let index = 0

  for (let i = 0; i < strs.length; i++) {
    let cur = strs[i].split('').sort().join('')
    if (!visited.has(cur)) {
      visited.set(cur, [])
    }
    const set = visited.get(cur)
    set.push(strs[i])
    visited.set(cur, set)
  }
  visited.forEach(item => {
    res.push(item)
  })
  return res
};
```

---

# 53. 最大子序和（简单）
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
```
示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
进阶:

如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
```
```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var maxSubArray = function (nums) {
  let max = nums[0]
  let sum = 0

  for (let i = 0; i < nums.length; i++) {
    let cur = nums[i]
    sum = sum > 0 ? sum + cur : cur
    max = Math.max(max, sum)
  }

  return max
};
```

---

# 62. 不同路径（中等）
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

示例1:
```
输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
```
示例2:
```
输入: m = 7, n = 3
输出: 28
```
```javascript
/**
 * @param {number} m
 * @param {number} n
 * @return {number}
 */
var uniquePaths = function (m, n) {
  const dp = Array(n).fill(Array(m).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (i === 0 || j === 0) {
        dp[i][j] = 1
      } else {
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
      }
    }
  }
  return dp[n - 1][m - 1]
};

```

---


# 64. 最小路径和（中等）
给定一个包含非负整数的 mxn网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例:
```
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```
```javascript
/**
 * @param {number[][]} grid
 * @return {number}
 */

var minPathSum = function (grid) {
  let [maxRow, maxCol] = [grid.length, grid[0].length]
  for (let row = 0; row < maxRow; row++) {
    for (let col = 0; col < maxCol; col++) {
      if (row === 0 && col !== 0) {
        grid[row][col] = grid[row][col] + grid[row][col - 1]
      }
      else if (row !== 0 && col === 0) {
        grid[row][col] = grid[row][col] + grid[row - 1][col]
      }
      else if (row !== 0 && col !== 0) {
        grid[row][col] = grid[row][col] + Math.min(grid[row][col - 1], grid[row - 1][col])
      }
    }
  }
  return grid[maxRow - 1][maxCol - 1]
};

```

---

# 66. 加一（简单）
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。
```
示例 1:

输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。
```
```
示例 2:

输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。
```
```javascript
/**
 * @param {number[]} digits
 * @return {number[]}
 */
var plusOne = function (digits) {
  let res = digits
  let len = res.length
  for (let i = len - 1; i >= 0; i--) {
    if (res[i] + 1 < 10) {
      res[i] = res[i] + 1
      return res
    } else {
      res[i] = 0
    }
  }
  digits.unshift(1)
  return digits
};
```

---

# 73. 矩阵置零（中等）
给定一个m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。

示例1:
```
输入: 
[
 [1,1,1],
 [1,0,1],
 [1,1,1]
]
输出: 
[
 [1,0,1],
 [0,0,0],
 [1,0,1]
]
```
示例2:
```
输入: 
[
 [0,1,2,0],
 [3,4,5,2],
 [1,3,1,5]
]
输出: 
[
 [0,0,0,0],
 [0,4,5,0],
 [0,3,1,0]
]
```
进阶:
```
一个直接的解决方案是使用 O(mn)的额外空间，但这并不是一个好的解决方案。
一个简单的改进方案是使用 O(m+n) 的额外空间，但这仍然不是最好的解决方案。
你能想出一个常数空间的解决方案吗？
```
```javascript
/**
 * 第一种，无脑遍历，带一丢丢优化（击败30%
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
var setZeroes = function (matrix) {
  let rowLen = matrix.length
  if (rowLen === 0) return []
  let colLen = matrix[0].length

  let colList = []

  for (let row = 0; row < rowLen; row++) {
    let rowFlag = false
    for (let col = 0; col < colLen; col++) {
      if (matrix[row][col] === 0) {
        rowFlag = true
        if (!colList.includes(col))
          colList.push(col)
      }
    }
    if (rowFlag) {
      matrix[row].fill(0)
    }
  }
  colList.forEach(col => {
    for (let row = 0; row < rowLen; row++) {
      matrix[row][col] = 0
    }
  })
};


/**
 * 第二种，用第一个来标记行列是否填充0（击败93%
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
var setZeroes = function (matrix) {
  let rowLen = matrix.length
  let colLen = matrix[0].length
  let flag = false
  for (let row = 0; row < rowLen; row++) {
    if (matrix[row][0] === 0) flag = true
    for (let col = 1; col < colLen; col++) {
      if (matrix[row][col] === 0) {
        matrix[0][col] = 0
        matrix[row][0] = 0
      }
    }
  }

  for (let row = 1; row < rowLen; row++) {
    for (let col = 1; col < colLen; col++) {
      if (matrix[0][col] === 0 || matrix[row][0] === 0) {
        matrix[row][col] = 0
      }
    }
  }

  if (matrix[0][0] === 0) {
    for (let col = 0; col < colLen; col++) {
      matrix[0][col] = 0
    }
  }

  if (flag) {
    for (let row = 0; row < rowLen; row++) {
      matrix[row][0] = 0
    }
  }
};


/**
 * 第三种，用集合记录下标，记录完后再遍历进行置0
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
var setZeroes = function (matrix) {
  const rowLen = matrix.length
  const colLen = matrix[0].length

  const rowList = new Set()
  const colList = new Set()

  let row = 0
  let col = 0
  for (row = 0; row < rowLen; row++) {
    for (col = 0; col < colLen; col++) {
      if (matrix[row][col] === 0) {
        rowList.add(row)
        colList.add(col)
      }
    }
  }
  for (row = 0; row < rowLen; row++) {
    for (col = 0; col < colLen; col++) {
      if (rowList.has(row) || colList.has(col)) {
        matrix[row][col] = 0
      }
    }
  }
};
```

---

# 78. 子集（中等）
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例:
```
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```
```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsets = function (nums) {
  let res = [[]];
  for (let i = 0; i < nums.length; i++) {
    res.map(e=>{
      res.push(e.concat(nums[i]))
    })
  }
  return res;
};
```

---
# 88. 合并两个有序数组（简单）
给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:

初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
示例:
```
输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```
```javascript
/**
 * 方法一: 偷工减料法
 *
 * @param {number[]} nums1
 * @param {number} m
 * @param {number[]} nums2
 * @param {number} n
 * @return {void} Do not return anything, modify nums1 in-place instead.
 */
var merge = function (nums1, m, nums2, n) {
  nums1.splice(m);
  nums1.push(...nums2);
  nums1.sort((a, b) => a - b);
};

/**
 * 方法二: 因为两个都是有序的，且nums1长度足够，那直
 * 接比较两个数组最大的一位，依次添加到nums1末尾即可
 */
var merge = function (nums1, m, nums2, n) {
  let total = m + n - 1,
    index1 = m - 1,
    index2 = n - 1;
  while (index1 >= 0 && index2 >= 0) 
    nums1[total--] = nums1[index1] > nums2[index2] ? nums1[index1--] : nums2[index2--];
  // 若nums2中仍有元素，则将他们填充到nums1前面
  // 若nums1中仍有元素，则可以不管，因为最后要得到的就是nums1
  while(index2>=0)
    nums1[total--]=nums2[index2--];
};
```
---

# 94. 二叉树的中序遍历（中等）
给定一个二叉树，返回它的中序遍历。

示例:
```
输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
```
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */

var inorderTraversal = function (root) {
  let res = []
  var backTrack = function (root, res) {
    if (root === null) {
      return
    }
    backTrack(root.left, res)
    res.push(root.val)
    backTrack(root.right, res)
  }
  backTrack(root, res)

  return res
};
```
---


# 102. 二叉树的层次遍历（中等）
给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

例如:
给定二叉树:[3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]

```
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function (root) {
  if (!root) return []
  let rootList = [root]
  const res = []
  while (rootList.length !== 0) {
    const valList = []
    const nodeList = []
    while (rootList.length !== 0) {
      const cur = rootList.shift()
      const left = cur.left
      const right = cur.right
      valList.push(cur.val)
      if (left) nodeList.push(left)
      if (right) nodeList.push(right)
    }
    rootList = nodeList
    res.push(valList)
  }
  return res
};


```
---

# 104. 二叉树的最大深度（简单）
给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明:叶子节点是指没有子节点的节点。
```
示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度3 。

```
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number}
 */
var maxDepth = function (root) {
  let res = 0
  if (!root) {
    return 0
  }
  else {
    res = Math.max(maxDepth(root.left), maxDepth(root.right)) + 1
  }
  return res
};


```
---

# 120. 三角形最小路径和（中等）
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
```
例如，给定三角形：

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为11（即，2+3+5+1= 11）。

说明：

如果你可以只使用 O(n)的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。
```
```javascript
/**
 * @param {number[][]} triangle
 * @return {number}
 */
var minimumTotal = function (triangle) {
  let len = triangle.length
  for (let row = 1; row < len; row++) {
    for (let col = 0; col < triangle[row].length; col++) {
      if (col === 0) {
        triangle[row][col] = triangle[row][col] + triangle[row - 1][col]
      } else if (col < row) {
        triangle[row][col] = triangle[row][col] + Math.min(triangle[row - 1][col - 1], triangle[row - 1][col])
      } else {
        triangle[row][col] = triangle[row][col] + triangle[row - 1][col - 1]
      }
    }
  }
  return Math.min(...triangle[len - 1])
};
```
---

# 121. 买卖股票的最佳时机（简单）
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。

注意你不能在买入股票前卖出股票。

示例 1:
```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

示例 2:
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

```
```javascript
/**
 * @param {number[]} prices
 * @return {number}
 */
var maxProfit = function(prices) {
    let min = prices[0];
    let max = 0;

    for (let i = 1; i < prices.length; i++){
      min = Math.min(min, prices[i]);
      max = Math.max(max, prices[i] - min);
    }

    return max;
};

console.log(maxProfit([7,6,4,3,1]))
```
---

# 122. 买卖股票的最佳时机II（简单）
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:
```
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```
示例 2:
```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```
示例 3:
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```
```javascript
/**
 * @param {number[]} prices
 * @return {number}
 */
var maxProfit = function(prices) {
    let max = 0;

    for (let i = 1; i < prices.length; i++)
      if (prices[i] > prices[i - 1])
        max = max - prices[i - 1] + prices[i];

    return max;
};

console.log(maxProfit([7,6,4,3,1]))
```


---

# 128. 最长连续序列（困难）
给定一个未排序的整数数组，找出最长连续序列的长度。

要求算法的时间复杂度为O(n)。

示例:
```
输入:[100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```
```javascript
/**
 * 若元素为连续序列的第一位，则判断，若一直满足连续
 * 则当前长度curLen + 1，若不满足连续，则记录当前的
 * 长度与最大的长度相比，更新最大长度，此外，最后返
 * 回的是当前获取的长度与最大长度之间的最大值，因为
 * 为了防止数组直接是连续的情况，例如[1,2,3,4,5]，此
 * 时curLlen是5，而res是1
 *
 * @param {number[]} nums
 * @return {number}
 */
var longestConsecutive = function (nums) {
  if (nums.length === 0) return 0
  let res = 1
  let curLen = 1
  nums = [...new Set(nums)].sort((a, b) => a - b)

  for (let i = 1; i < nums.length; i++) {
    let pre = nums[i - 1]
    let cur = nums[i]
    if (cur === pre + 1) {
      curLen += 1
    } else {
      res = Math.max(res, curLen)
      curLen = 1
    }
  }
  return Math.max(res, curLen)
};


/**
 * 方法二，利用集合来做，循环遍历集合，若集合中没有
 * 当前元素的前一位，则可以将当前元素当为序列的开头
 * 然后利用循环，若集合有当前元素的后一位，则长度+1
 * 重复此步骤
 *
 * @param {number[]} nums
 * @return {number}
 */
var longestConsecutive = function(nums) {
  if (nums.length < 2) return nums.length
  let set = new Set(nums)
  let max = 0
  for (let i of set.keys()) {
    if (!set.has(i - 1)) {
      let temp = 1,
        cur = i
      while (set.has(cur + 1)) {
        temp++
        cur++
      }
      max = Math.max(max, temp)
    }
  }
  return max
};
```


---

# 131. 分隔回文串（中等）
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。
```
示例:

输入:"aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]

```
```javascript
/**
 * @param {string} s
 * @return {string[][]}
 */
var partition = function (s) {
  let res = []
  let len = s.length
  let path = []
  backTrace(s, 0, len, path, res)
  return res
};

var backTrace = function (str, start, len, path, res) {
  if (start === len) {
    res.push([...path])
    return
  }

  for (let i = start; i < len; i++) {
    let cur = str.slice(start, i + 1)
    console.log(`start: ${start}\ti: ${i}\t\tcur: ${cur}`)
    if (isPartition(cur)) {
      path.push(cur)
      backTrace(str, i + 1, len, path, res)
      path.pop()
    }
  }
}

var isPartition = function (str) {
  let i = 0;
  let j = str.length - 1;
  while (i < j) {
    if (str[i++] != str[j--]) return false;
  };
  return true;
}
```


---

# 136. 只出现一次的数字（简单）
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:
```
输入: [2,2,1]
输出: 1
```
示例 2:
```
输入: [4,1,2,1,2]
输出: 4
```
```javascript
/**
 * 方法一: 投机取巧，因为只有一个数字是出现一次的，所以直接
 * 在数组中找此元素的第一个和最后一个索引，比较一下
 *
 * @param {number[]} nums
 * @return {number}
 */
var singleNumber = function(nums) {
  for(let item of nums){
    if(nums.indexOf(item)===nums.lastIndexOf(item))
      return item;
  }
};
/**
 * 方法二: 直接使用异或来处理，相同为0，不同为1，异或同一个* 数两次，原数不变。
 */
var singleNumber = function(nums) {
  let res;
  for(let item of nums)
    res=res^item;
  return res;
};
```

---

# 139. 单词拆分（中等）
给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定s 是否可以被空格拆分为一个或多个在字典中出现的单词。
```
说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
    注意你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

```
```javascript
/**
 * 动态规划解法, 用dp[i]记录字符串s[0:i]是否可以由字典中的单词构成。
两重for循环遍历s[i:j], 对于特定的i和j， 有三种情况:

如果i == 0, 这种情况很简单，我们遍历一下，看看s[i:j]是否在字典中即可。
对于i > 0的情况，我们要分两种情况讨论, 因为如果s[0:i]无法由字典中的单词构成，那么即使s[i:j]可以由字典中的单词构成，也毫无意义。
2.1 如果s[0:i]无法由字典中的单词构成，那么这种case可以直接忽略。
2.2. 如果s[0:i]可以由字典中的单词构成，同时s[i:j]是字典中的单词，那么我们可以认为s[0:j]可以由字典中的单词构成。将dp[j]置为1
最后，检查dp[len(s)]即可.
 *
 * @param {string} s
 * @param {string[]} wordDict
 * @return {boolean}
 */
var wordBreak = function (s, wordDict) {
  let len = s.length
  let wordSet = new Set(wordDict)
  let res = Array(len + 1).fill(false)
  res[0] = true
  for (let start = 0; start < len; start++) {
    for (let end = start + 1; end <= len; end++) {
      let cur = s.slice(start, end)
      if (!res[start]) break
      if (wordSet.has(cur)) {
        res[end] = true
      }

      /**
       * 为什么不这样呢，以为若前面的子串已经不能由wordDict组成，就可以break跳过了，而不需要继续进行循环，可以节约很多时间
       * 
       * if (res[start] && wordSet.has(cur)) {
       *  res[end] = treu
       * } 
       * 
       */
    }
  }
  return res[len]
};
```

---


# 141. 环形链表（简单）
给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
```

示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。


示例2：

输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。


示例 3：

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。


进阶：

你能用 O(1)（即，常量）内存解决此问题吗？

```
```javascript
/**
 * 方法一: 投机取巧，因为只有一个数字是出现一次的，所以直接
 * 在数组中找此元素的第一个和最后一个索引，比较一下
 *
 * @param {number[]} nums
 * @return {number}
 */
var singleNumber = function(nums) {
  for(let item of nums){
    if(nums.indexOf(item)===nums.lastIndexOf(item))
      return item;
  }
};
/**
 * 方法二: 直接使用异或来处理，相同为0，不同为1，异或同一个* 数两次，原数不变。
 */
var singleNumber = function(nums) {
  let res;
  for(let item of nums)
    res=res^item;
  return res;
};
```

---

# 152. 乘积最大的子序列（中等）
给定一个整数数组 nums，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。

示例 1:
```
输入: [2,3,-2,4]
输出: 6
解释:子数组 [2,3] 有最大乘积 6。
示例 2:

输入: [-2,0,-1]
输出: 0
解释:结果不能为 2, 因为 [-2,-1] 不是子数组。
```
```javascript
/**
 * 循环遍历数组，定义两个变量max,min，如果当前元素为负值
 * 则与前面取得的最大值相乘之后，最大的变最小，最小的变
 * 最大，所以我们交换max与min的值，然后继续获取当前最大
 * 与最小
 * 
 * @param {number[]} nums
 * @return {number}
 */
var maxProduct = function (nums) {
  let len = nums.length
  let max = min = 1
  let res = nums[0]
  for (let i = 0; i < len; i++) {

    let cur = nums[i]
    if (cur < 0) {
      [max, min] = [min, max]
    }
    max = Math.max(max * cur, cur)
    min = Math.min(min * cur, cur)

    res = Math.max(max, res)
  }
  return res
};
```

---

# 169. 求众数（简单）
给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在众数。

示例 1:
```
输入: [3,2,3]
输出: 3
```
示例 2:
```
输入: [2,2,1,1,1,2,2]
输出: 2
```
```javascript
/**
 * 方法一: 愚蠢的办法，找出各元素的次数再进行比较。
 *
 * @param {number[]} nums
 * @return {number}
 */
var majorityElement = function (nums) {
  let obj = {},
    maxIndex = 0;
  for (let item of nums) {
    obj[item] = obj[item] ? obj[item] + 1 : 1;
  }
  for (let i in obj) {
    if (obj[i] === Math.max(...Object.values(obj)))
      return i
  }
};

/**
 * 方法二: 众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
 */
var majorityElement = function (nums) {
  let arr = nums;
  arr.sort((a, b) => a - b);
  return arr[Math.ceil(arr.length / 2)];
};

/**
 * 方法三: 摩尔投票法。
 */
var majorityElement = function (nums) {
  let count = 0,
    res = 0;
  for (let item of nums) {
    if (count === 0) res = item;;
    if (item != res) count -= 1;
    else count += 1;
  }
  return res;
};
```

---

# 198. 打家劫舍（简单）
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
```
示例 1:

输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
    偷窃到的最高金额 = 1 + 3 = 4 。
示例 2:

输入: [2,7,9,3,1]
输出: 12
解释: 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
    偷窃到的最高金额 = 2 + 9 + 1 = 12 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/house-robber
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var rob = function (nums) {
  if (nums.length === 0) return 0
  let arr = [nums[0], Math.max(nums[0], nums[1])]
  for (let i = 2; i < nums.length; i++) {
    arr[i] = Math.max(arr[i - 2] + nums[i], arr[i - 1])
  }
  return arr[nums.length - 1]
};

```

---

# 202. 快乐数（简单）
编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

示例:
```
输入: 19
输出: true
```
解释: 
```
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```
```javascript
/**
 * 如果不是快乐数，他会陷入循环4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4，所以判断是否为4即可跳出
 *
 * @param {number} n
 * @return {boolean}
 */
var isHappy = function (n) {
  let next = n
  while (true) {
    let num = next
    next = 0
    while (num !== 0) {
      const last = num % 10
      next += last ** 2
      num = Math.floor(num / 10)
    }
    if (next === 1) return true
    if (next === 4) return false
  }
};
```

---



# 206. 反转链表（简单）
反转一个单链表。
```
示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？

```
```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var reverseList = function (head) {
  let res = null
  let p = head
  while (p) {
    [p.next, res, p] = [res, p, p.next]
  }
  return res
};
```

---

# 234. 回文链表（简单）
请判断一个链表是否为回文链表。
```
示例 1:

输入: 1->2
输出: false
示例 2:

输入: 1->2->2->1
输出: true
进阶：
你能否用O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/palindrome-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {boolean}
 */
var isPalindrome = function (head) {
  let stack = []
  while (head) {
    stack.push(head.val)
    head = head.next
  }
  let start = 0
  let end = stack.length - 1
  while (start < end) {
    if (stack[start] !== stack[end]) return false
    start++
    end--
  }
  return true
};
```

---

# 236. 二叉树的最近公共祖先（中等）
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

 例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]  
 
![image](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

示例 1:
```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```
示例2:
```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```

说明:
```
所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉树中。

```
```javascript
/**
 * 以root为根节点，查找是否有p节点或者q节点，如果都有则返回root本身，即为所求，若左右只有一个，则返回左节点或者右节点，
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
var lowestCommonAncestor = function (root, p, q) {
  if (root === null) return null
  if (root === p || root === q) {
    return root
  }
  const left = lowestCommonAncestor(root.left, p, q)
  const right = lowestCommonAncestor(root.right, p, q)
  if (left && right) return root
  return left || right
};
```


---



# 237. 删除链表中的节点（简单）
请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。

现有一个链表 -- head = [4,5,1,9]，它可以表示为:
 
![image](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/01/19/237_example.png)

示例 1:
```
输入: head = [4,5,1,9], node = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```
示例 2:
```
输入: head = [4,5,1,9], node = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
 ```

说明:
```
链表至少包含两个节点。
链表中所有节点的值都是唯一的。
给定的节点为非末尾节点并且一定是链表中的一个有效节点。
不要从你的函数中返回任何结果。
```
```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} node
 * @return {void} Do not return anything, modify node in-place instead.
 */
var deleteNode = function(node) {
    node.val=node.next.val;
    node.next=node.next.next;
};
```


---


# 238. 除自身外数组的乘积（中等）
给定长度为n的整数数组nums，其中n > 1，返回输出数组output，其中 output[i]等于nums中除nums[i]之外其余各元素的乘积。

示例:
```
输入: [1,2,3,4]
输出: [24,12,8,6]
说明: 请不要使用除法，且在O(n) 时间复杂度内完成此题。

进阶：
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）

```
```javascript
/**
 * @param {number[]} nums
 * @return {number[]}
 */

var productExceptSelf = function (nums) {
  let len = nums.length
  let left = Array(len).fill(1)
  let right = [...left]
  let res = []
  for (let start = 1; start < len; start++) {
    let end = len - start - 1
    left[start] = nums[start - 1] * left[start - 1]
    right[end] = nums[end + 1] * right[end + 1]
  }
  for (let i = 0; i < len; i++) {
    res.push(left[i] * right[i])
  }
  return res
};
```


---



# 240. 搜索二维矩阵 II（中等）
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
示例:

现有矩阵 matrix 如下：
```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```
给定 target = 5，返回 true。

给定 target = 20，返回 false。
```javascript
/**
 * 方法一: 每一行每一列都是递增的，那么我们可以以右上角
 * 为起点，右上角的值比目标值大就左移，比目标值小就下移
 *
 * @param {number[][]} matrix
 * @param {number} target
 * @return {boolean}
 */
var searchMatrix = function (matrix, target) {
  if (matrix.length === 0 || matrix[0].length === 0)
    return false

  let row = 0,
    col = matrix[0].length - 1,
    rowLength = matrix.length,
    current;
  while (row < rowLength && col >= 0) {
    current = matrix[row][col];
    if (current === target) return true;
    else if (current > target) col -= 1;
    else row += 1;
  }
  return false;
};
```

---

---


# 242. 有效的字母异位词（简单）
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
```
示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
说明:
你可以假设字符串只包含小写字母。

进阶:
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？
```
```javascript
/**
 * @param {string} s
 * @param {string} t
 * @return {boolean}
 */
var isAnagram = function (s, t) {
  let obja = {

  }
  let objb = {

  }
  if (s.length !== t.length) {
    return false
  }

  let a = s.split('')
  let b = t.split('')

  for (let i = 0; i < a.length; i++) {
    obja[a[i]] = obja[a[i]] ? obja[a[i]] + 1 : 1
  }

  for (let i = 0; i < b.length; i++) {
    objb[b[i]] = objb[b[i]] ? objb[b[i]] + 1 : 1
  }

  for (let i in obja) {
    if (obja[i] !== objb[i])
      return false
  }
  return true
};
```

# 268. 缺失数字（简单）
给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。

示例 1:
```
输入: [3,0,1]
输出: 2
```
示例 2:
```
输入: [9,6,4,2,3,5,7,0,1]
输出: 8
```
```javascript
/**
 * 求和相减就是要的数了
 *
 * @param {number[]} nums
 * @return {number}
 */
var missingNumber = function (nums) {
  let len = nums.length;
  let sum = nums.reduce((pre, cur) => pre + cur);
  return len * (len + 1) / 2 - sum;
};
```

---

# 283. 移动零（简单）
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
```
示例:

输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:

必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
```
```javascript
/**
 * 求和相减就是要的数了
 *
 * @param {number[]} nums
 * @return {number}
 */
var missingNumber = function (nums) {
  let len = nums.length;
  let sum = nums.reduce((pre, cur) => pre + cur);
  return len * (len + 1) / 2 - sum;
};
```

---

# 287. 寻找重复数（中等）
给定一个包含n + 1 个整数的数组nums，其数字都在 1 到 n之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

示例 1:
```
输入: [1,3,4,2,2]
输出: 2
示例 2:

输入: [3,1,3,4,2]
输出: 3
说明：

不能更改原数组（假设数组是只读的）。
只能使用额外的 O(1) 的空间。
时间复杂度小于 O(n2) 。
数组中只有一个重复的数字，但它可能不止重复出现一次。

```
```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var findDuplicate = function (nums) {
  // let count = Array.from({ length: nums.length }, _ => 0)
  let count = Array(nums.length).fill(0)
  for (let start = 0; start < nums.length / 2; start++) {
    let end = nums.length - start - 1
    count[nums[start]] += 1
    count[nums[end]] += 1
    if (count[nums[start]] > 1) {
      return nums[start]
    }
    if (count[nums[end]] > 1) {
      return nums[end]
    }
  }
};
```

---

# 334. 递增的三元序列（中等）
给定一个未排序的数组，判断这个数组中是否存在长度为 3 的递增子序列。

数学表达式如下:

如果存在这样的i, j, k,且满足0 ≤ i < j < k ≤ n-1，
使得arr[i] < arr[j] < arr[k] ，返回 true ;否则返回 false 。
说明: 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1) 。

示例 1:
```
输入: [1,2,3,4,5]
输出: true
示例 2:

输入: [5,4,3,2,1]
输出: false
```
```javascript
/**
 * 创建两个变量start，end，若当前元素小于等于start
 * 则更新start，若当前元素小于等于end，则更新end
 * 若之后，当前元素比end还大，那就证明存在递增序列
 *
 * @param {number[]} nums
 * @return {boolean}
 */
var increasingTriplet = function (nums) {
  let start = Number.MAX_VALUE, end = Number.MAX_VALUE
  for (let i = 0; i < nums.length; i++) {
    let cur = nums[i]
    if (cur <= start) {
      start = cur
    }
    else if (cur <= end) {
      end = cur
    }
    else {
      return true
    }
  }
  return false
};
```

---

# 338. 比特位记数（中等）
给定一个非负整数num。对于0 ≤ i ≤ num 范围中的每个数字i，计算其二进制数中的 1 的数目并将它们作为数组返回。

示例 1:
```
输入: 2
输出: [0,1,1]
示例2:

输入: 5
输出: [0,1,1,2,1,2]
```
```javascript
/**
 * @param {number} num
 * @return {number[]}
 */
var countBits = function (num) {
  let res = [0]
  let n = 0
  for (let i = 1; i <= num; i++) {
    let max = Math.pow(2, n)
    if (i < max) {
      res[i] = 1 + res[i - max / 2]
    }
    else {
      res[i] = 1
      n += 1
    }
  }
  return res
};
```

---


# 347. 前K个高频元素（中等）
给定一个非空的整数数组，返回其中出现频率前k高的元素。

示例 1:
```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```
示例 2:
```
输入: nums = [1], k = 1
输出: [1]
```
说明：
```
你可以假设给定的k总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
你的算法的时间复杂度必须优于 O(n log n) ,n是数组的大小。
```
```javascript
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
var topKFrequent = function (nums, k) {
  const map = new Map()
  nums.forEach(num => {
    const count = map.get(num)
    map.set(num, count ? count + 1 : 1)
  })
  const arr = []
  for (let item of map) {
    arr.push(item)
  }
  arr.sort((a, b) => b[1] - a[1])
  const res = []
  for (let i = 0; i < k; i++) {
    res.push(arr[i][0])
  }
  return res
};
```

---

# 350. 两个数组的交集（简单）
给定两个数组，编写一个函数来计算它们的交集。
```
示例 1:

输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2,2]
```
```
示例 2:

输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [4,9]
```
```
说明：

输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
我们可以不考虑输出结果的顺序。
进阶:

如果给定的数组已经排好序呢？你将如何优化你的算法？
如果 nums1 的大小比 nums2 小很多，哪种方法更优？
如果 nums2 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？
```
```javascript
/**
 * 方法一：第二个数组找到就删除
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
var intersect = function (nums1, nums2) {
  let res = []

  if (nums1.length < nums1.length) {
    for (let i = 0; i < nums1.length; i++) {
      let index = nums2.indexOf(nums1[i])
      if (index !== -1) {
        res.push(nums1[i])
        nums2.splice(index, 1)
      }
    }
  } else {
    for (let i = 0; i < nums2.length; i++) {
      let index = nums1.indexOf(nums2[i])
      if (index !== -1) {
        res.push(nums2[i])
        nums1.splice(index, 1)
      }
    }
  }

  return res
};

/**
 * 方法二：排序之后慢慢找
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
var intersect = function (nums1, nums2) {
  nums1.sort((a, b) => a - b)
  nums2.sort((a, b) => a - b)

  let one = 0, two = 0, res = []

  while (one < nums1.length && two < nums2.length) {
    if (nums1[one] < nums2[two]) {
      one += 1
    } else if (nums1[one] > nums2[two]) {
      two += 1
    } else {
      res.push(nums1[one])
      one += 1
      two += 1
    }
  }
  return res
};
```

---

# 378. 有序矩阵中第K小的元素（中等）
给定一个n x n矩阵，其中每行和每列元素均按升序排序，找到矩阵中第k小的元素。
请注意，它是排序后的第k小元素，而不是第k个元素。

示例:
```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

返回 13。
```
```javascript
/**
 * @param {number[][]} matrix
 * @param {number} k
 * @return {number}
 */
var kthSmallest = function (matrix, k) {
  let res = []

  var flat = function (arr) {
    arr.forEach(item => {
      if (Array.isArray(item)) {
        return flat(item)
      } else {
        res.push(item)
      }
    })
    return res
  }
  return flat(matrix).sort((a, b) => a - b)[k - 1]
};
```

---


# 461. 汉明距离（简单）
两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。

给出两个整数 x 和 y，计算它们之间的汉明距离。
```
注意：
0 ≤ x, y < 231.

示例:

输入: x = 1, y = 4

输出: 2

解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

上面的箭头指出了对应二进制位不同的位置。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/hamming-distance
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
```javascript
/**
 * @param {number} x
 * @param {number} y
 * @return {number}
 */
var hammingDistance = function (x, y) {
  let xBinary = binary(x)
  let yBinary = binary(y)
  let maxLen = Math.max(xBinary.length, yBinary.length)
  xBinary = xBinary.padStart(maxLen, '0')
  yBinary = yBinary.padStart(maxLen, '0')
  let count = 0
  for (let i = 0; i < maxLen; i++) {
    let cur = xBinary[i]
    if (yBinary[i] !== cur) {
      count += 1
    }
  }
  return count
};

var binary = function (num) {
  let res = ''
  while (num !== 0) {
    res = num % 2 + res
    num = parseInt(num / 2)
  }
  return res
}
```

---

# 477. 汉明距离总和（中等）
两个整数的汉明距离 指的是这两个数字的二进制数对应位不同的数量。

计算一个数组中，任意两个数之间汉明距离的总和。
```
示例:

输入: 4, 14, 2

输出: 6

解释: 在二进制表示中，4表示为0100，14表示为1110，2表示为0010。（这样表示是为了体现后四位之间关系）
所以答案为：
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/total-hamming-distance
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
```javascript
/**
 * 正常的方法遍历会超时，横着遍历不行，那就竖着看，比较每一位上有多少个不同，
 * 加起来就是要的答案，例如有4个数，根据规律，不同的数是1的数量乘上剩下的数量
 *
 * @param {number[]} nums
 * @return {number}
 */

var totalHammingDistance = function (nums) {
  nums = nums.map(item => binary(item).padStart(32, 0))
  let len = nums.length
  let count = 0
  for (let i = 0; i < 32; i++) {
    let sum = 0
    for (let j = 0; j < nums.length; j++) {
      let cur = nums[j][i]
      if (cur === '1') {
        sum += 1
      }
    }
    count += sum * (len - sum)
  }
  return count
};

var binary = function (num) {
  let res = ''
  while (num !== 0) {
    res = num % 2 + res
    num = parseInt(num / 2)
  }
  return res
}
```

---

# 521. 最长特殊序列I（简单）
给定两个字符串，你需要从这两个字符串中找出最长的特殊序列。最长特殊序列定义如下：该序列为某字符串独有的最长子序列（即不能是其他字符串的子序列）。

子序列可以通过删去字符串中的某些字符实现，但不能改变剩余字符的相对顺序。空序列为所有字符串的子序列，任何字符串为其自身的子序列。

输入为两个字符串，输出最长特殊序列的长度。如果不存在，则返回 -1。

示例 :
```
输入: "aba", "cdc"
输出: 3
解析: 最长特殊序列可为 "aba" (或 "cdc")
```
说明:
```
两个字符串长度均小于100。
字符串中的字符仅含有 'a'~'z'。
```
```javascript
/**
 * 如果两个字符串相等则返回-1，因为题目说空序列是所有
 * 的子序列，那就说明只要是不相等的字符串，那么就一定
 * 存在一个最长特殊序列，因为可以通过删减，所以返回长
 * 度最大的即可
 * 
 * @param {string} a
 * @param {string} b
 * @return {number}
 */
var findLUSlength = function(a, b) {
  let lenA = a.length, lenB = b.length;
  return lenA === lenB ? -1 : Math.max(a.length,b.length);
};

```

---

# 575. 分糖果（简单）
给定一个偶数长度的数组，其中不同的数字代表着不同种类的糖果，每一个数字代表一个糖果。你需要把这些糖果平均分给一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。

示例 1:
```
输入: candies = [1,1,2,2,3,3]
输出: 3
解析: 一共有三种种类的糖果，每一种都有两个。
     最优分配方案：妹妹获得[1,2,3],弟弟也获得[1,2,3]。这样使妹妹获得糖果的种类数最多。
```
示例 2 :
```
输入: candies = [1,1,2,3]
输出: 2
解析: 妹妹获得糖果[2,3],弟弟获得糖果[1,1]，妹妹有两种不同的糖果，弟弟只有一种。这样使得妹妹可以获得的糖果种类数最多。
```
```javascript
/**
 * 方法一: 挺简单的，只需要比较一下，数组的一半与种类数
 * 谁大谁小即可，如果数组的一半比种类数大，那就可以得到
 * 种类数的糖果，如果小，则只能得到数组一半的糖果。
 *
 * 例如：[1,1,2,2,3,3,3,3],这里可以最多拿到3种糖果
 *       [1,1,2,3],这里最多可以拿到2种糖果
 * 
 * @param {number[]} candies
 * @return {number}
 */
var distributeCandies = function (candies) {
  // 这里的type是糖果的种类数。
  let type = new Set(candies).size;
  
  return candies.length / 2 > type ? type : candies.length / 2;
};
```

---

# 617. 合并二叉树（简单）
给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为NULL 的节点将直接作为新二叉树的节点。
```
示例1:

输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

```
```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} t1
 * @param {TreeNode} t2
 * @return {TreeNode}
 */
var mergeTrees = function (t1, t2) {
  if (!t1) return t2
  if (!t2) return t1
  t1.val += t2.val
  t1.left = mergeTrees(t1.left, t2.left)
  t1.right = mergeTrees(t1.right, t2.right)
  return t1
};

```

---

# 674. 最长连续递增序列（简单）
给定一个偶数长度的数组，其中不同的数字代表着不同种给定一个未经排序的整数数组，找到最长且连续的的递增序列。
```
示例 1:

输入: [1,3,5,4,7]
输出: 3
解释: 最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。 
示例 2:

输入: [2,2,2,2,2]
输出: 1
解释: 最长连续递增序列是 [2], 长度为1。
注意：数组长度不会超过10000。

```
```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var findLengthOfLCIS = function (nums) {
  if (nums.length === 0) return 0
  let curLen = 1
  let max = 1

  for (let i = 1; i < nums.length; i++) {
    let pre = nums[i - 1]
    let cur = nums[i]
    if (cur > pre) {
      curLen += 1
    } else {
      curLen = 1
    }
    max = Math.max(max, curLen)
  }
  return max
};
```

---

# 733. 图像渲染（简单）
有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

示例 1:
```
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。
```
注意:
```
image 和 image[0] 的长度在范围 [1, 50] 内。
给出的初始点将满足 0 <= sr < image.length 和 0 <= sc < image[0].length。
image[i][j] 和 newColor 表示的颜色值在范围 [0, 65535]内。
```
```javascript
/**
 * @param {number[][]} image
 * @param {number} sr
 * @param {number} sc
 * @param {number} newColor
 * @return {number[][]}
 */
var floodFill = function (image, sr, sc, newColor, val) {
  const visit = {};
  const fill = (img, row, col, color, val) => {
    if (row < 0 || col < 0 || row >= img.length || col >= img[0].length) return null;
    if (!visit[`${row}${col}`] && img[row][col] === val) {
      visit[`${row}${col}`] = true;
      img[row][col] = color;
      fill(img, row, col + 1, color, val);
      fill(img, row, col - 1, color, val);
      fill(img, row + 1, col, color, val);
      fill(img, row - 1, col, color, val);
    }
  }
  fill(image, sr, sc, newColor, image[sr][sc]);
  return image;
};
```

---

# 771. 宝石与石头（简单）
给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。

J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

示例 1:
```
输入: J = "aA", S = "aAAbbbb"
输出: 3
```
示例 2:
```
输入: J = "z", S = "ZZ"
输出: 0
```
注意:
```
S 和 J 最多含有50个字母。
 J 中的字符不重复。
```
```javascript
/**
 * @param {string} J
 * @param {string} S
 * @return {number}
 */
var numJewelsInStones = function(J, S) {
    let count = 0;
    S.split("").forEach(item=>{
      if (J.includes(item))
        count += 1;
    })
    return count;
};

console.log(numJewelsInStones("aA","aAAbbbb"))
```

---

# 804. 唯一摩尔斯密码词（简单）
国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。

为了方便，所有26个英文字母对应摩尔斯密码表如下：

`[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]`
给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。例如，"cab" 可以写成 "-.-..--..."，(即 "-.-." + "-..." + ".-"字符串的结合)。我们将这样一个连接过程称作单词翻译。

返回我们可以获得所有词不同单词翻译的数量。

例如:
```
输入: words = ["gin", "zen", "gig", "msg"]
输出: 2
解释: 
各单词翻译如下:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

共有 2 种不同翻译, "--...-." 和 "--...--.".
 ```

注意:
```
单词列表words 的长度不会超过 100。
每个单词 words[i]的长度范围为 [1, 12]。
每个单词 words[i]只包含小写字母。
```
```javascript
/**
 * @param {string[]} words
 * @return {number}
 */
var uniqueMorseRepresentations = function (words) {
  const code = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."];
  const res = [];
  words.forEach(word => {
    let item = "";
    word.split("").forEach(letter => {
      item += code[letter.charCodeAt() - 97];
    })
    res.push(item)
  })
  return new Set(res).size;
};
```

---

# 806. 柠檬水找零（简单）
在柠檬水摊上，每一杯柠檬水的售价为 5 美元。

顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。

每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。

注意，一开始你手头没有任何零钱。

如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

示例 1：
```
输入：[5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。
```
示例 2：
```
输入：[5,5,10]
输出：true
```
示例 3：
```
输入：[10,10]
输出：false
```
示例 4：
```
输入：[5,5,10,10,20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 false。
 ```

提示：
```
0 <= bills.length <= 10000
bills[i] 不是 5 就是 10 或是 20 
```
```javascript
/**
 * 这题需要注意一点，如果遇到需要找零15元的情况，应先选择
 * 找零10 + 5的组合，而不是选择5 + 5 + 5的组合，因为5块钱
 * 什么情况都可以成功找零，而十块钱只适用于更少的情况。
 *
 * @param {number[]} bills
 * @return {boolean}
 */
var lemonadeChange = function (bills) {
  const change = {
    "5": 0,
    "10": 0,
    "20": 0,
  };
  for (let i = 0; i < bills.length; i++) {
    change[bills[i]]++;
    let give = bills[i] - 5;
    switch (give) {
      case 0: break;
      case 5:
        if (change[5] >= 1) {
          change[5]--;
          break;
        }
        else {
          return false;
        }
      case 10:
        if (change[5] >= 2) {
          change[5] -= 2;
          break;
        }
        else if (change[10] >= 1) {
          change[10]--;
          break;
        }
        else {
          return false;
        }
      case 15:
        if (change[10] >= 1 && change[5] >= 1) {
          change[5]--;
          change[10]--;
          break;
        }
        else if (change[5] >= 3) {
          change[5] -= 3;
          break;
        }
        else {
          return false;
        }
    }
  }
  return true;
};
```


---

# 807. 保持城市天际线（中等）
在二维数组grid中，grid[i][j]代表位于某处的建筑物的高度。 我们被允许增加任何数量（不同建筑物的数量可能不同）的建筑物的高度。 高度 0 也被认为是建筑物。

最后，从新数组的所有四个方向（即顶部，底部，左侧和右侧）观看的“天际线”必须与原始数组的天际线相同。 城市的天际线是从远处观看时，由所有建筑物形成的矩形的外部轮廓。 请看下面的例子。

建筑物高度可以增加的最大总和是多少？

例子：
```
输入： grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
输出： 35
解释： 
The grid is:
[ [3, 0, 8, 4], 
  [2, 4, 5, 7],
  [9, 2, 6, 3],
  [0, 3, 1, 0] ]

从数组竖直方向（即顶部，底部）看“天际线”是：[9, 4, 8, 7]
从水平水平方向（即左侧，右侧）看“天际线”是：[8, 7, 9, 3]

在不影响天际线的情况下对建筑物进行增高后，新数组如下：

gridNew = [ [8, 4, 8, 7],
            [7, 4, 7, 7],
            [9, 4, 8, 7],
            [3, 3, 3, 3] ]
            ```
            
说明:
```
1 < grid.length = grid[0].length <= 50。
 grid[i][j] 的高度范围是： [0, 100]。
一座建筑物占据一个grid[i][j]：换言之，它们是 1 x 1 x grid[i][j] 的长方体。
```
```javascript
/**
* 只用无脑遍历每一个元素，如果不是他所在行与列中的最大值
* 的话，那就让这个元素变成行列最大值中最小的那个，因为要
* 保持天际线不变，意思就是不能超过行列的最大值。
*
* @param {number[][]} grid
* @return {number}
*/
var maxIncreaseKeepingSkyline = function (grid) {
  let rowMax = []; // 用来存每一行的最大值
  let colMax = []; // 用来存每一列的最大值
  
  /*
    这样求行列最大值也行，但是耗时间内存
    let rowMax = grid.map(row => Math.max(...row))
    let colMax = grid[0].map((col, colIndex) => Math.max(...grid.map(row => row[colIndex])));
  */
  
  let sum = 0;
  for (let row = 0; row < grid.length; row++) {
    rowMax.push(Math.max(...grid[row]));
  }
  for (let col = 0; col < grid[0].length; col++) {
    let maxItem = grid[0][col];
    for (let row = 1; row < grid.length; row++) {
      if (grid[row][col] > maxItem) {
        maxItem = grid[row][col];
      }
    }
    colMax.push(maxItem);
  }
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[0].length; col++) {
      if (grid[row][col] !== rowMax[row] && grid[row][col] !== colMax[col]) {
        sum += Math.min(rowMax[row], colMax[col]) - grid[row][col];
        // grid[row][col] = Math.min(rowMax[row], colMax[col]);
      }
    }
  }
  return sum;
};
```

# 832. 反转图像（简单）
给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。

水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。

反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。

示例 1:
```
输入: [[1,1,0],[1,0,1],[0,0,0]]
输出: [[1,0,0],[0,1,0],[1,1,1]]
解释: 首先翻转每一行: [[0,1,1],[1,0,1],[0,0,0]]；
     然后反转图片: [[1,0,0],[0,1,0],[1,1,1]]
```
示例 2:
```
输入: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
输出: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
解释: 首先翻转每一行: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]]；
     然后反转图片: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
```
说明:
```
1 <= A.length = A[0].length <= 20
0 <= A[i][j] <= 1
```

```javascript
/**
 * @param {number[][]} A
 * @return {number[][]}
 */
var flipAndInvertImage = function (A) {
  for (let i = 0; i < A.length; i++) {
    A[i] = A[i].reverse()
  }
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[0].length; j++) {
      A[i][j] = A[i][j] === 0 ? 1 : 0;
    }
  }
  return A;
};
```

---

# 887. 掉落的鸡蛋（困难）
你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。

每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。

你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。

每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。

你的目标是确切地知道 F 的值是多少。

无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？

> 提示：
1 <= K <= 100
1 <= N <= 10000

示例 1：
```
输入：K = 1, N = 2
输出：2
解释：
鸡蛋从 1 楼掉落。如果它碎了，我们肯定知道 F = 0 。
否则，鸡蛋从 2 楼掉落。如果它碎了，我们肯定知道 F = 1 。
如果它没碎，那么我们肯定知道 F = 2 。
因此，在最坏的情况下我们需要移动 2 次以确定 F 是多少。
```
示例 2：
```
输入：K = 2, N = 6
输出：3
```
示例 3：
```
输入：K = 3, N = 14
输出：4
 ```

> tips: https://segmentfault.com/a/1190000016364846#articleHeader6  
https://github.com/Shellbye/Shellbye.github.io/issues/42

一个蛋跟一层楼的情况已经可以确定了哦  
![image](https://segmentfault.com/img/remote/1460000016364856)

---

```javascript
/**
 * 方法一: 因为我们是要找无论 F的初始值如何的条件下的
 * 最少查找次数，所以我们要分上下两种情况，找最大值。
 * （此方法LeetCode上超出时间限制）
 
 * @param {number} K
 * @param {number} N
 * @return {number}
 */
var superEggDrop = function (K, N) {
  if (K === 1 || N === 1 ) {
    return N;
  }
  let minCount = N;
  for (let floor = 1; floor <= N; floor += 1) {
    // 如果蛋碎了，那就往下找，蛋数K记得要 - 1，楼高是刚才的高度floor - 1，
    // 如果蛋没碎，那就网上找，蛋数K不变，楼高是总楼高N - 刚才的高度floor。

    // 因为无法确定蛋会碎的楼层F，所以我们要比较一下网上找与往下找两者之间的最大值，以确保能够确定F层的位置。
    let count = Math.max(superEggDrop(K - 1, floor - 1), superEggDrop(K, N - floor));
    // count别忘了 + 1，因为循环不包括第一次丢鸡蛋，要把他的次数也加上。
    minCount = Math.min(minCount, count + 1);
  }
  return minCount;
};

/**
 * 方法二: 原理同上，但用了对象来存取已经计算过的次数。
 * 减少了递归循环的次数。
 */
let data = {};

var superEggDrop = function (K, N) {
  if (K === 1) {
    return N;
  }
  if (N === 0) {
    return 0;
  }
  
  // 不可以用data[`${K}${N}`]，LeetCode会报错，中间加个小横线就过了
  if (data[`${K}-${N}`]) {
    return data[`${K}-${N}`];
  }

  let start = 1,
    end = N;

  while (start < end) {
    let floor = Math.ceil((start + end) / 2),
      lower = superEggDrop(K - 1, floor - 1),
      higher = superEggDrop(K, N - floor);
    if (lower > higher) {
      end = floor - 1;
    } else if (lower < higher) {
      start = floor + 1;
    } else {
      start = end = floor;
    }
  }

  const minCount = 1 + Math.min(
    Math.max(superEggDrop(K - 1, start - 1), superEggDrop(K, N - start)),
    Math.max(superEggDrop(K - 1, end - 1), superEggDrop(K, N - end))
  );
  data[`${K}-${N}`] = minCount;
  return minCount;
};

/**
 * 方法三: 原理同上，用二分法来做，但是LeetCode不知为何
 * 说解答错误，本地答案明明正确。
 */
let data = {};

var superEggDrop = function (K, N) {
  if (K === 1) {
    return N;
  }
  if (N === 0) {
    return 0;
  }
  if (data[`${K}${N}`]) {
    return data[`${K}${N}`];
  }

  let start = 1,
    end = N;

  while (start < end) {
    let floor = Math.ceil((start + end) / 2);
    let lower = superEggDrop(K - 1, floor - 1),
      higher = superEggDrop(K, N - floor);
    if (lower > higher) {
      end = floor - 1;
    } else if (lower < higher) {
      start = floor + 1;
    } else {
      start = end = floor;
    }
  }

  let minCount = 1 + Math.min(
    Math.max(superEggDrop(K - 1, start - 1), superEggDrop(K, N - start)),
    Math.max(superEggDrop(K - 1, end - 1), superEggDrop(K, N - end))
  );
  data[`${K}${N}`] = minCount;
  return data[`${K}${N}`];
};

/**
 * 方法四: 转换思维，在K个蛋时候已经确定了多少层不是F，
 * 如果K个蛋已经确认的层数大于楼层数，则已经找到楼层F了。
 */
var superEggDrop = function (K, N) {
  let count = [];
  for (let i = 0; i < K + 1; i += 1) {
    count.push(new Array(N + 1).fill(0))
  }

  /**
   * count[egg][move]指的是在蛋数为egg移动次数为move时候
   * 能确定到的楼层数量。
   */
  for (let move = 1; move <= N; move += 1) {
    for (let egg = 1; egg <= K; egg += 1) {
      // count[egg][move] = 没碎测得的楼层数 + 碎了之后测得的楼层数 + 当前楼层数1
      count[egg][move] = count[egg][move - 1] + count[egg - 1][move - 1] + 1;
      if (count[egg][move] >= N) {
        return move;
      }
    }
  }
};

/**
 * 方法五: 原理同上，但简化过
 */
var superEggDrop = function (K, N) {
  let count = Array.from({
    length: K + 1
  }).fill(0);
  let minCount = 0;
  while (count[K] < N) {
    minCount += 1;
    for (let egg = K; egg > 0; egg -= 1) {
      count[egg] += count[egg - 1] + 1;
    }
  }
  return minCount;
};
```

---

# 938. 二叉搜索树的范围和（简单）
给定二叉搜索树的根结点 root，返回 L 和 R（含）之间的所有结点的值的和。

二叉搜索树保证具有唯一的值。

 
示例 1：
```
输入：root = [10,5,15,3,7,null,18], L = 7, R = 15
输出：32
```
示例 2：
```
输入：root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
输出：23
 ```

提示：
```
树中的结点数量最多为 10000 个。
最终的答案保证小于 2^31。
```

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * 方法一：BFS
 *
 * @param {TreeNode} root
 * @param {number} L
 * @param {number} R
 * @return {number}
 */
var rangeSumBST = function (root, L, R) {
  let res = 0
  const loop = (root, L, R) => {
    if (!root) return null;

    if (root.val >= L && root.val <= R) {
      res += root.val;
    }
    if (root.val > L) {
      loop(root.left, L, R)
    }
    if (root.val < R) {
      loop(root.right, L, R)
    }
  }

  loop(root, L, R)
  return res;
};

/**
 * 方法二：中序遍历
 *
 * @param {TreeNode} root
 * @param {number} L
 * @param {number} R
 * @return {number}
 */
var rangeSumBST = function (root, L, R) {
  const dfs = (node) => {
    if (!node) return 0;
    if (node.val < L) {
      return dfs(node.right);
    }
    if (node.val > R) {
      return dfs(node.left);
    }
    return node.val + dfs(node.left) + dfs(node.right);
  }
  return dfs(root);
};
```

---

# 977. 有序数组的平方（简单）
给定一个按非递减顺序排序的整数数组 A，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。

 

示例 1：
```
输入：[-4,-1,0,3,10]
输出：[0,1,9,16,100]
```
示例 2：
```
输入：[-7,-3,2,3,11]
输出：[4,9,9,49,121]
 ```

提示：
```
1 <= A.length <= 10000
-10000 <= A[i] <= 10000
A 已按非递减顺序排序。
```
```javascript
/**
 * @param {number[]} A
 * @return {number[]}
 */
var sortedSquares = function(A) {
  let arr = [];
  A.forEach(item=>{
    arr.push(item ** 2)
  })
  return arr.sort((a,b)=>a-b)
};
```


---
# 1021. 删除最外层的括号（简单）
有效括号字符串为空 ("")、"(" + A + ")" 或 A + B，其中 A 和 B 都是有效的括号字符串，+ 代表字符串的连接。例如，""，"()"，"(())()" 和 "(()(()))" 都是有效的括号字符串。

如果有效字符串 S 非空，且不存在将其拆分为 S = A+B 的方法，我们称其为原语（primitive），其中 A 和 B 都是非空有效括号字符串。

给出一个非空有效字符串 S，考虑将其进行原语化分解，使得：S = P_1 + P_2 + ... + P_k，其中 P_i 是有效括号字符串原语。

对 S 进行原语化分解，删除分解中每个原语字符串的最外层括号，返回 S 。

 

示例 1：
```
输入："(()())(())"
输出："()()()"
解释：
输入字符串为 "(()())(())"，原语化分解得到 "(()())" + "(())"，
删除每个部分中的最外层括号后得到 "()()" + "()" = "()()()"。
```
示例 2：
```
输入："(()())(())(()(()))"
输出："()()()()(())"
解释：
输入字符串为 "(()())(())(()(()))"，原语化分解得到 "(()())" + "(())" + "(()(()))"，
删除每隔部分中的最外层括号后得到 "()()" + "()" + "()(())" = "()()()()(())"。
```
示例 3：
```
输入："()()"
输出：""
解释：
输入字符串为 "()()"，原语化分解得到 "()" + "()"，
删除每个部分中的最外层括号后得到 "" + "" = ""。
```

提示：
```
S.length <= 10000
S[i] 为 "(" 或 ")"
S 是一个有效括号字符串
```
```javascript
/**
 * 简单的操作栈，左括号就入栈，右括号就出栈，如果栈空，则
 * 代表外层的括号已经消除，则把中间的内容加进字符串。
 * 
 * @param {string} S
 * @return {string}
 */
var removeOuterParentheses = function (S) {
  const arr = S.split("");
  let stack = 0;
  let s = "";
  let start = 0;
  let end = 0;
  let flag = true;
  for (let i = 0; i < arr.length; i++) {
    if (stack > 0 && flag) {
      start = i;
      flag = false;
    }
    if (arr[i] === '(') {
      stack++;
    }
    if (arr[i] === ')') {
      stack--;
    }
    if (stack=== 0) {
      end = i;
      flag = true;
      s += arr.slice(start, end).join("");
    }
  }
  return s;
};

```

# 1202. 交换字符串中的元素
给你一个字符串s，以及该字符串中的一些「索引对」数组pairs，其中pairs[i] =[a, b]表示字符串中的两个索引（编号从 0 开始）。

你可以 任意多次交换 在pairs中任意一对索引处的字符。

返回在经过若干次交换后，s可以变成的按字典序最小的字符串。

示例 1:
```
输入：s = "dcab", pairs = [[0,3],[1,2]]
输出："bacd"
解释： 
交换 s[0] 和 s[3], s = "bcad"
交换 s[1] 和 s[2], s = "bacd"
```
示例 2：
```
输入：s = "dcab", pairs = [[0,3],[1,2],[0,2]]
输出："abcd"
解释：
交换 s[0] 和 s[3], s = "bcad"
交换 s[0] 和 s[2], s = "acbd"
交换 s[1] 和 s[2], s = "abcd"
```
示例 3：
```
输入：s = "cba", pairs = [[0,1],[1,2]]
输出："abc"
解释：
交换 s[0] 和 s[1], s = "bca"
交换 s[1] 和 s[2], s = "bac"
交换 s[0] 和 s[1], s = "abc"
```

提示：
```
1 <= s.length <= 10^5
0 <= pairs.length <= 10^5
0 <= pairs[i][0], pairs[i][1] <s.length
s中只含有小写英文字母
```
```javascript
/**
 * @param {number} n
 * @param {number[][]} connections
 * @return {number}
 */
var smallestStringWithSwaps = function (s, pairs) {
  /* 找上级 */
  var unionSearch = function (root, pre) {
    let init = root
    while (root !== pre[root]) {
      root = pre[root]
    }
    /* 路径压缩 */
    while (init !== root) {
      [pre[init], init] = [root, pre[init]]
    }
    return root
  }

  let pre = new Array(s.length)
  for (let i = 0; i < s.length; i++) {
    pre[i] = i
  }

  /* 连通起来 */
  for (let i = 0; i < pairs.length; i++) {
    const [start, end] = pairs[i]
    const root1 = unionSearch(start, pre)
    const root2 = unionSearch(end, pre)
    if (root1 !== root2) {
      pre[root1] = root2
    }
  }
  const groups = {}

  for (let i = 0; i < pre.length; i++) {
    pre[i] = unionSearch(pre[i], pre)
    /* 将同一个上级的元素分组放进对象里 */
    if (!groups[pre[i]]) {
      groups[pre[i]] = []
    }
    groups[pre[i]].push(i)
  }

  let ans = s.split('')
  Object.values(groups).forEach(group => {
    /* 返回一个当前分组对应的字符串 */
    let series = group.map(index => ans[index])
    /* 排排序找到最大的情况 */
    let maxString = series.sort((a, b) => a.localeCompare(b))
    for (let i = 0; i < group.length; i++) {
      /* 遍历分组内的下标，将原字符串当前下标的值替换成最大字符串的值 */
      ans[group[i]] = maxString[i]
    }
  })

  return ans.join('')
};
```

---


# 1254. 统计封闭岛屿的数目(周赛)
有一个二维矩阵 grid ，每个位置要么是陆地（记号为 0 ）要么是水域（记号为 1 ）。

我们从一块陆地出发，每次可以往上下左右 4 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「岛屿」。

如果一座岛屿 完全 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「封闭岛屿」。

请返回封闭岛屿的数目。
```
示例 1：

输入：grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
输出：2
解释：
灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。
示例 2：



输入：grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
输出：1
示例 3：

输入：grid = [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,1,1,1,0,1],
             [1,0,1,0,1,0,1],
             [1,0,1,1,1,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1]]
输出：2
 

提示：

1 <= grid.length, grid[0].length <= 100
0 <= grid[i][j] <=1

```
```javascript
/**
 * @param {number[][]} grid
 * @return {number}
 */
var closedIsland = function (grid) {
  let res = 0
  let rowLen = grid.length
  let colLen = grid[0].length
  let visited = Array.from({ length: rowLen }, _ => Array.from({ length: colLen }, __ => false))
  let pos = [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }]

  var dfs = function (row, col) {
    if (row === 0 || row === rowLen - 1 || col === 0 || col === colLen - 1) return false

    if (visited[row][col]) return true
    visited[row][col] = true
    let flag = true
    for (let i = 0; i < 4; i++) {
      const { x, y } = pos[i]
      if (grid[row + x][col + y] === 0) {
        if (!dfs(row + x, col + y)) flag = false
      }
    }
    return flag
  }

  for (let row = 0; row < rowLen; row++) {
    for (let col = 0; col < colLen; col++) {
      if (!visited[row][col] && grid[row][col] === 0) {
        if (dfs(row, col)) res++
      }
    }
  }
  return res
};
```

---


# 1319. 连通网络的操作次数（中等）(周赛)
用以太网线缆将n台计算机连接成一个网络，计算机的编号从0到n-1。线缆用connections表示，其中connections[i] = [a, b]连接了计算机a和b。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回-1 。

示例 1：

![image](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/11/sample_1_1677.png)
```
输入：n = 4, connections = [[0,1],[0,2],[1,2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
```
示例 2：

![image](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/11/sample_2_1677.png)
```
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
输出：2
```
示例 3：
```
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2]]
输出：-1
解释：线缆数量不足。
```
示例 4：
```
输入：n = 5, connections = [[0,1],[0,2],[3,4],[2,3]]
输出：0
```

提示：
```
1 <= n <= 10^5
1 <= connections.length <= min(n*(n-1)/2, 10^5)
connections[i].length == 2
0 <= connections[i][0], connections[i][1]< n
connections[i][0] != connections[i][1]
没有重复的连接。
两台计算机不会通过多条线缆连接。
```

```javascript
/**
 * @param {number} n
 * @param {number[][]} connections
 * @return {number}
 */
var makeConnected = function (n, connections) {
  /* 如果线不够，直接返回-1，因为如果线够，通过一定方法无论如何都可以全部连通 */
  if (connections.length < n - 1) return -1

  var unionSearch = function (root, pre) {
    let init = root
    while (root !== pre[root]) {
      root = pre[root]
    }
    /* 路径压缩 */
    while (init !== root) {
      [pre[init], init] = [root, pre[init]]
    }
    return root
  }

  /* 总共有多少个未连通的计算机 */
  let res = n
  let pre = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    pre[i] = i
  }

  for (let i = 0; i < connections.length; i++) {
    const [start, end] = connections[i]
    let root1 = unionSearch(start, pre)
    let root2 = unionSearch(end, pre)
    /* 如果未连通，则连通起来，未连通数量-1 */
    if (root1 !== root2) {
      pre[root1] = root2
      res--
    }
  }
  return res - 1
};
```

---