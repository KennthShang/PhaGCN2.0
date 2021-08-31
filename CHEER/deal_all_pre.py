f = open("all_pre.fasta","r")
g = open("all_simple_pre.fasta","w")
for each in f:
    each = each.replace("\n","")
    if ">" in each:
        (x,y) = each.split(" ",1)
        if "," in y:
            (y,z) = y.split(",",1)
        y = y.replace("/"," ")
        y = y.replace("-", " ")
        y = y.replace("[", " ")
        y = y.replace("]", " ")
        if " " not in y:
            print("label error,please revise",each)
        each = x +" "+y
    print(each,file=g)