import re,collections
def main(fn):
    t=open(fn).read().lower()
    w=re.findall(r"[a-z0-9]+",t)
    c={x:t.count(x) for x in set(w)}
    for k,v in sorted(c.items(),key=lambda x:x[1],reverse=True)[:10]:
        print(k,v)
if __name__=="__main__":main("example.txt")
