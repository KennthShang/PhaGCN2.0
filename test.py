f = open("ALL.contigs.fa","r")
g = open("deal.fa","w")
for each in f:
    each = each.replace("\n","")
    if ">" not in each:
        each = each.replace("n","")
        each = each.replace("N","")
    print(each,file=g)