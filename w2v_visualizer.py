# encoding: utf-8
import sys, os
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

def visualize(model, output_path):
    meta_file = "w2v_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 150))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            if word == '':
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2v_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2v_metadata'
    embed.metadata_path = meta_file
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2v_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to visualize result on tensorboard'.format(output_path))

if __name__ == "__main__":
    """
    run `python w2v_visualizer.py word2vec.model visualize_result`
    """
    try:
        model_path = sys.argv[1]
        output_path  = sys.argv[2]
    except:
        print("python w2v_visualizer.py word2vec.model visualize_result ")
    #model = Word2Vec.load(model_path)
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=False)
    visualize(model, output_path)
