from PIL import Image
import numpy as np

def LNIP_Feature_Extract(roi):
    row1 = ''.join(str(roi[0]))
    row1 = row1[1:len(row1)-1]

    row2 = ''.join(str(roi[1]))
    row2 = row1[1:len(row1)-1]

    row3 = ''.join(str(roi[2]))
    row3 = row1[1:len(row1)-1]
    
    i6,i7,i8 = row1.split()
    i5,Ic,i1 = row2.split()
    i4,i3,i2 = row3.split()
    
    signs = []
    magn = []
    sign_stri = ""
    mag_stri= ""

    nei_i1 = [i7,i8,i2,i3]
    nei_i2 = [i1,i3]
    nei_i3 = [i1,i2,i4,i5]
    nei_i4 = [i3,i5]
    nei_i5 = [i3,i4,i6,i7]
    nei_i6 = [i5,i7]
    nei_i7 = [i5,i6,i8,i1]
    nei_i8 = [i7,i1]
    all_nei = [i1,i2,i3,i4,i5,i6,i7,i8]

    indices = {'i1':i1,'i2':i2,'i3':i3,'i4':i4,'i5':i5,'i6':i6,'i7':i7,'i8':i8}
    neigh_lists  = {'ne_i1':nei_i1,'ne_i2':nei_i2,'ne_i3':nei_i3,'ne_i4':nei_i4,'ne_i5':nei_i5,'ne_i6':nei_i6,'ne_i7':nei_i7,'ne_i8':nei_i8}

    def B_1_i(nei_lis,compare_element):
        bli = ''
        for i in nei_lis:
            if(int(i)<int(compare_element)):
                bli+='0'
            elif(int(i)>=int(compare_element)):
                bli+='1'
            else:
                pass
        return bli

    def B_2_i(nei_lis,centre_element):
        b2i = ''
        for i in nei_lis:
            if(int(i)<int(centre_element)):
                b2i+='0'
            elif(int(i)>=int(centre_element)):
                b2i+='1'
            else:
                pass
        return b2i

    def mags(neis,comp):
        m_sum = 0.0
        for k in neis:
            m_sum+=abs((int(k)-int(comp)))
        return float(m_sum/len(neis))

    def thresholds(alls,centre_ele):
        thre_sum = 0.0
        for h in alls:
            thre_sum+=abs(int(h)-int(centre_ele))
        return float(thre_sum/8)

    for_ind = list(indices.keys())
    for_ind.sort()
    for_nei = list(neigh_lists.keys())
    for_nei.sort()

    for one,two in zip(for_ind,for_nei):
        res1 = B_1_i(neigh_lists[two],indices[one])
        res2 = B_2_i(neigh_lists[two],Ic)
        res3 = int(res1,2)^int(res2,2)
        #print str(bin(res3)[2:].zfill(4))+'  '+str(indices[one])
        D = bin(res3)[2:].count('1')
        M = len(neigh_lists[two])
        if(D>=int((M/2))):
            signs.append(str(1))
        else:
            signs.append(str(0))

    for one,two in zip(for_ind,for_nei):
        Mi = mags(neigh_lists[two],indices[one])
        Tc = thresholds(all_nei,Ic)
        if(Mi>=Tc):
            magn.append(str(1))
        else:
            magn.append(str(0))
    sign_stri = sign_stri.join(signs)
    mag_stri = mag_stri.join(magn)
    return "Sign Value :"+str(int(sign_stri,2))+"    "+"Magnitude Value: "+str(int(mag_stri,2))

temp = Image.open('trail.png') #Give the name of the image you want to use
img = np.array(temp)
#print img
img_shape = img.shape

size = 3 #window size i.e. here is 3x3 window

shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
strides = 2 * img.strides
patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
patches = patches.reshape(-1, size, size)
print "The number of 3x3 Windows in the Image  "+str(len(patches))+"\n"

for roi in patches:
    res = LNIP_Feature_Extract(roi)
    print res

