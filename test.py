import os

if __name__ =="__main__":
    result = ""
    num = 0
    for line in open('./filter.txt').readlines():
        num += 1
        line = line.strip()
        result += "'" + line + "',"
        if num % 10 == 0:
            result += "\n"
    print(result)