import streamlit as stm
from tensorflow.keras.models import load_model
import numpy as np 
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
html_temp1 = """
    <div style="background-color:black;padding:4px">
    <p style="color:white;text-align:center;">ABOUT</p>
    </div>
    <p>--------------------------------------------------</p>
    <p> This is a web app of my project smart predictor which predicts the next words based on the previous words you write.
    For full detail kindly visit my blog. <p><a href="https://medium.com/analytics-vidhya/smart-composer-with-attention-mechanism-b67e798803b3" target="_blank">Smart Predictor</a></p>
    <p style="font-weight: bold;">How to run this web app</p>
    """
html_temp2 = """
    <p>Instructions:</p>
    <p>1. Running time will be slow sometime as it is deployed on free version so it takes time takes to load data and there is a bug of stream lit that whenever user input is taken it reloads the whole  page</p>
    <p>2. when you use gmail your conversation with user stored in database so it suggest you the word based on that but here i used enron email dataset for training so i stored conversation of all the user.</p>
    """


def main():
    html_temp = """
    <div style="background-color:#042958;padding:10px">
    <h2 style="color:white;text-align:center;">SMART PREDICTOR WEB APP</h2>
    </div>
    <p>--------------------------------------------------------------------------------------------------------------------</p>
    """
    stm.sidebar.markdown(html_temp1,unsafe_allow_html=True)
    vid_file  = open("fullyfinal.mp4","rb").read()
    stm.sidebar.video(vid_file)
    stm.sidebar.markdown(html_temp2,unsafe_allow_html=True)



    encoder_model , decoder_model ,x_tr, y_tr, x_val,y_val,x_tokenizer,y_tokenizer,max_inp_len,max_out_len = load_data()
    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index
    input_word_index= x_tokenizer.word_index
    stm.markdown(html_temp,unsafe_allow_html=True)
    stm.text("Please write some generic message for better result")
    html_temp6 = """
    <p>These are the message you can try which is the conversation between two person.<a href="https://shrib.com/?v=nc#Gemsbok_M0mL1w"> Link</a> </p>
    

    """
    stm.markdown(html_temp6,unsafe_allow_html=True)

    message = stm.text_area("Type any message")
    final = sentence_conv(message,x_tokenizer,y_tokenizer,max_inp_len)
    stm.button("Predict")
    result  = bring_my_sentence(final,encoder_model,decoder_model,target_word_index,reverse_target_word_index,max_out_len)
    stm.success(result)
    stm.header("Final Text:   "+message+result)
    stm.header("Why smart predictor is useful?")
    html_temp3 = """
    <p>Smart Predictor is useful in many ways one of the main thing which make smart predictor powerful is it saves time . For eg. you want to write any review suppose your given rating matches with another user so while writing review predictor suggest you words so you don't have to write all the words.</p>
    <p> 2. When you visit doctor and tell your symtomps if your symptoms matches with another person. Doctor saves his time as he don't have to write similar medicine again and again.</p>

    """
    
    html_temp4 = """
    <style>
    .fa {
  padding: 20px;
  font-size: 50px;
  width: 50px;
  text-align: center;
  text-decoration: none;
  margin: 5px 2px;
}

</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<br>
<p>Connect with me on:</p>
<p>Linkedin<a href="https://www.linkedin.com/in/aman-varyani-885725181/" class="fa fa-linkedin"></a>Github<a href="https://github.com/amanv1906" class="fa fa-github"></a></p>

    """
    stm.markdown(html_temp3,unsafe_allow_html=True)
    stm.markdown(html_temp4,unsafe_allow_html=True)

def sentence_conv(text,x_tokenizer,y_tokenizer,max_inp_len):
    p=preprocess(text)
    x_tr_seq = x_tokenizer.texts_to_sequences(p)
    x_check   =   pad_sequences(x_tr_seq,  maxlen=max_inp_len, padding='post')
    return(x_check)
        


#it is used to clean puncuation from the string 
def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, '')
    return text

#preprocess the data
def preprocess(data):
    output = []
    punct = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
    pline= clean_special_chars(data.lower(), punct)
    output.append(pline)
    return output
class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

@stm.cache
def load_data():
    encoder_model = load_model("my_last_model.h5", custom_objects={'AttentionLayer': AttentionLayer})
    decoder_model = load_model("my_last_decoder_model.h5",custom_objects={'AttentionLayer': AttentionLayer})
    with open("train.pickle", 'rb') as f:
        x_tr, y_tr, x_val,y_val,x_tokenizer,y_tokenizer,max_inp_len,max_out_len = pickle.load(f)
    return encoder_model , decoder_model, x_tr, y_tr, x_val,y_val,x_tokenizer,y_tokenizer,max_inp_len,max_out_len
#THIS function predicts the next sentence
def bring_my_sentence(input_seq,encoder_model,decoder_model,target_word_index,reverse_target_word_index,max_out_len):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eos'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eos'  or len(decoded_sentence.split()) >= (max_out_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

if __name__ == '__main__':
	main()







    
