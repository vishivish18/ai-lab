
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import tensorflow as tf


# In[2]:


x_data = np.linspace(0.0,10.0,1000000)


# In[3]:


noise = np.random.randn(len(x_data))


# In[4]:


noise.shape


# In[6]:


# y = mx+b
#b =5



# In[7]:


y_true = (0.5 * x_data) + 5 + noise


# In[8]:


x_df = pd.DataFrame(data =x_data, columns=['X Data'])


# In[9]:


y_df = pd.DataFrame(data = y_true, columns=['Y'])


# In[12]:


x_df.head()


# In[13]:


my_data = pd.concat([x_df,y_df], axis=1)


# In[14]:


my_data.head()


# In[16]:


my_data.sample(n=250).plot(kind='scatter',x='X Data', y='Y')


# In[17]:


batch_size = 8


# In[19]:


np.random.randn(2)


# In[18]:


m = tf.Variable(1.14)


# In[20]:


b = tf.Variable(0.6)


# In[21]:


xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])


# In[22]:


y_model = m*xph + b


# In[23]:


error = tf.reduce_sum(tf.square(yph-y_model))


# In[24]:


optimizr = tf.train.GradientDescentOptimizer(learning_rate=0.001)


# In[25]:


train = optimizr.minimize(error)


# In[26]:


init = tf.global_variables_initializer()


# In[38]:


with tf.Session() as sess:
    sess.run(init)
    
    batches = 20000
    for i in range(batches):
        ran_ind = np.random.randint(len(x_data), size=batch_size)
        
        feed = {xph:x_data[ran_ind], yph:y_true[ran_ind]}
        
        sess.run(train,feed_dict=feed)
        
    model_m, model_b = sess.run([m,b])


# In[39]:


model_m


# In[40]:


model_b


# In[30]:


y_hat = x_data*model_m + model_b


# In[33]:


my_data.sample(250).plot(kind='scatter',x ='X Data',y='Y')
plt.plot(x_data,y_hat,'r')


# In[73]:


feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]


# In[74]:


estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3,random_state=101)


# In[77]:


x_train.shape


# In[78]:


x_eval.shape


# In[79]:


input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)


# In[80]:


train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)


# In[86]:


eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)


# In[87]:


estimator.train(input_fn=input_func,steps=1000)


# In[88]:


train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


# In[89]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)


# In[91]:


print('TRANING DATA METRICS')
print(train_metrics)


# In[92]:


print('EVAL DATA METRICS')
print(eval_metrics)


# In[95]:


brand_new_data = np.linspace(0,10,10)


# In[96]:


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)


# In[97]:


estimator.predict(input_fn=input_fn_predict)


# In[98]:


list(estimator.predict(input_fn=input_fn_predict))


# In[99]:


predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])


# In[100]:


predictions


# In[101]:


my_data.sample(250).plot(kind='scatter',x='X Data', y='Y')
plt.plot(brand_new_data,predictions,'r*')


# In[ ]:




