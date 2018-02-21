
# coding: utf-8

# # my first python experience

# In[3]:


1+2


# In[4]:


for i in range(5):
    print(i)


# In[5]:


get_ipython().magic(u'matplotlib inline')


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20)
y = x**2

plt.plot(x,y)


# # 数据类型

# In[8]:


x = 3.0
print type(x)


# In[9]:


print (x + 1)


# In[10]:


print (x**2)


# In[12]:


print(x**3)


# In[13]:


t = True
f = False
print(t or f)


# In[14]:


print(t == f)


# In[15]:


h ='%s %s %d' %(t,f,12)
print(h)


# In[16]:


xs = [3,1,2]
print (xs[-1])


# In[19]:


xs.append('you')
print (xs)


# In[22]:


x = xs.pop()
print x,xs


# In[23]:


print(xs[:-1])


# In[24]:


xs.append('love')
print xs


# In[25]:


print xs[-1]


# In[26]:


print xs[:-1]


# In[27]:


for xss in xs:
    print xss


# In[31]:


for idx, xss in enumerate(xs):
    print 'the %d element is %s' %(idx+1,xss)


# # 字典

# In[32]:


d = {'cat':'cute','dog':'furry'}
print d['cat']


# In[34]:


for animal in d:
    feature = d[animal]
    print 'the  feature of %s is %s'%(animal,feature)


# In[35]:


#访问对应的键与对应的值，使用iteritems
for x,y in d.iteritems():
    print 'the %s is %s'%(x,y)


# In[36]:


print 'fish' in xs


# In[37]:


print len(xs)


# In[38]:


#总结：[]是列表，列表中的元素是由顺序的，而在{}里的是集合，集合是没有顺序的。
#元组是一个有序列表，不可以改变顺序，元组可以在字典中作键，还可以作为集合的元素
d = {(x,x+1):x for x in range(10)}
print d


# In[39]:


t =(7,8)
print d[t]


# # 函数

# In[43]:


def sign(x):
    if x >0:
        return 'POS'
    elif x<0:
        return 'NEG'
    else:
        return '0'
    
for x in [-1, 0, 2]:
    print sign(x)


# # Numpy

# In[44]:


import numpy as np
a = np.array([1,2,3])
print type(a)
print a.shape


# In[45]:


a[0]=5
print a


# In[49]:


b = np.array([[1,3,5],[2,4,6]])
print b
print b.shape
print b[1,1]
print b[np.arange(2),[0,2]] #在每行中选择对应的列号的元素


# In[52]:


bool_idx = (b >2)
print bool_idx
print b[bool_idx]


# In[53]:


x  = np.array([1,2],dtype=np.float64)
print x.dtype
print x


# In[55]:


y = np.array([5,6],dtype=np.float64)
print x+y
print np.add(x,y)


# In[57]:


print x.dot(b) #矩阵相乘
print np.dot(x,b)


# In[63]:


print b
print np.sum(b,axis=0)
print np.sum(b,axis=1)
print b.T  #转置


# # 广播

# In[73]:


m = [1,0,1]
mm=np.empty_like(b)
#mmm=np.tile(m,(2,1)) #复制mm，行数为复制两次，列数复制一次，
#有了广播机制可以不用此行
print m+b


# In[78]:


w = np.array([4,5])
print  np.reshape(w,(2,1))
print  b+np.reshape(w,(2,1))


# # SciPy

# In[80]:


from scipy.misc import imread, imsave, imresize
img = imread('C:/Users/Myy/Pictures/783320543fd992db.jpg')
print img.dtype,img.shape


# In[82]:


img_1 = imresize(img,(80,80))           #将图像大小改变


# In[89]:


plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_1) #matplotlib.pyplot 中的imshow可以用于显示图像


# # matplotlib

# In[86]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,3*np.pi,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sinx')
plt.legend(['sin'])
plt.show()

