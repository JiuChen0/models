import numpy as np
import tensorflow as tf
from models import MyModel_TransLTEE  # Please replace with your actual model
from config import Config
import time
from sklearn.model_selection import train_test_split

def main():
    '''Define Time(t0) to Evaluate Effect'''
    t0=40
    config = Config()

    '''Load and process the data'''
    TY = np.loadtxt('./data/IHDP/csv/ihdp_npci_' + "1" + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]
    out_treat = np.loadtxt('./data/IHDP/Series_y_' + "1" + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    ys = out_treat[:, 1:(t0 + 2)]
    x, X_test, y, y_test, t, t_test = train_test_split(matrix, ys, ts, test_size=0.2)
    
    '''Load models'''
    model = MyModel_TransLTEE(t0,t)
    model_CF = MyModel_TransLTEE(t0,1-t)
    n_train = tf.cast(x.shape[0],tf.int32)
    print(x.shape, y.shape, t.shape, n_train)
    # input_shape = [[n_train,dim_x],[n_train,dim_y-1],[n_train,dim_y-1]]
    # model.build(input_shape = input_shape)
    # model_CF.build(input_shape = input_shape)
    # model.get_weights()

    print('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION...')
    print('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')
    print('TRAINING MODEL...')

    '''Define OPtimizer and Functions'''
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.lr, decay=config.weight_decay)
    sched = tf.keras.optimizers.schedules.ExponentialDecay(config.lr, decay_steps=3, decay_rate=0.5)
    loss_func = tf.keras.metrics.Mean(name='loss_func')
    accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_func')

    '''For each train step:
    Predict the fauctal & counterfactural outcomes 
    -> Evaluate the predicted error and Wasserstain distance as loss function
    -> Gradient Descent'''
    def train_step(X_train, tar_train, tar_real, gamma = 1e-4):
        with tf.GradientTape() as tape:
            predictions, predict_error, distance = model(
                X_train, tar_train, tar_real)
            
            CF_predictions, _, _ = model_CF(
                X_train, tar_train, tar_real)
            
            pred_effect = tf.reduce_mean(abs(predict_error-CF_predictions),axis=0)
            print(pred_effect)
            loss = predict_error + gamma * distance
            # loss_groundtruth = train_loss(abs(CF_predictions), groundtruth)
            # loss = predict_error + loss_groundtruth + gama*distance

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_func(loss)
        tar_real = tf.expand_dims(tar_real,-1)
        accuracy_func(tar_real, predictions)
        # accuracy_func(pred_effect, groundtruth)
        # print([CF_predictions])

    '''Training'''
    for epoch in range(100):
        loss_func.reset_states()
        accuracy_func.reset_states()
        start = time.time()

        train_step(x, y[:,:-1], y[:,1:])

        # Timing and printing happens here after each epoch
        epoch_time = time.time() - start
        print(f'Epoch {epoch + 1} Loss {loss_func.result():.4f} Accuracy {accuracy_func.result():.4f}')
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')

    print('MODEL SUCCESSFULLY TRAINED!')



if __name__ == '__main__':
    main()