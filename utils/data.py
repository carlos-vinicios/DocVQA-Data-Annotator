
def data_remover(xarray, ind):
    removed_count = 0
    for i in ind:
        del xarray[i - removed_count]
        removed_count +=1
    return xarray