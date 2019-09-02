import numpy as np
import tensorflow as tf
from alexnet import AlexNet

class PredictImage():
    
    def __init__(self):
        self.model_file_path = 'snaps/tensorflow_snaps/models/model_epoch65.ckpt'
        self.num_classes = 250
        self.batch_size = 1
        self.x = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        self.model = AlexNet(self.x, self.keep_prob, self.num_classes, [])
        self.score = self.model.fc8
        self.saver = tf.train.Saver()
        
        # Restrict GPU memory
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction=0.5
        
        # Create session
        sess = tf.Session(config = self.config)
        sess.run(tf.global_variables_initializer())
        
        # Load TensorFlow model
        self.saver.restore(sess, self.model_file_path)
        
        self.sess = sess

    def Predict(self, image_): 
        """
        This function calculates and returns top-5 predictions of the CNN model
        image_ : numpy array, image of user interface
        """
        with tf.device('/cpu:0'):
            # Create data object and iterator
            val_data = self.CreateDataset(image_)    
            iterator = tf.data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
            next_batch = iterator.get_next()

        validation_init_op = iterator.make_initializer(val_data)
        sess = self.sess
        sess.run(validation_init_op)
        img_batch = sess.run(next_batch)
        
        # Calculate probabilities
        score = sess.run([self.score], feed_dict={self.x: img_batch, self.keep_prob: 1.})
        score_np = np.array(score)
        
        # Return top-5 predictions
        top_5_prediction = tf.nn.top_k(score_np, k = 5, sorted=True)
        top_5 = sess.run(top_5_prediction)
        return top_5

    def CreateDataset(self, image_):
        """
        This function creates TensorFlow Dataset object using input image
        image_ : numpy array, image of user interface
        """
        data = tf.data.Dataset.from_tensors(image_)
        data = data.map(self.Arrange_image)
        data = data.batch(self.batch_size)
        return data
            
    def Arrange_image(self, images):
        """
        This is used in dataset mapping
        """
        img_resized = tf.image.resize_images(images, [227, 227])
        img_centered = tf.subtract(img_resized, tf.constant([123.68, 116.779, 103.939], dtype=tf.float32))
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr
            