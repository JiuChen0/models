import numpy as np
import tensorflow as tf
from models import MyModel_TransLTEE  # Please replace with your actual model
from config import Config
import time
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import train_test_split

def main():
    # Load and process the data
    t0=40
    config = Config()
    
    
    # for j in range(1, 11):
    TY = np.loadtxt('./data/IHDP/csv/ihdp_npci_' + "1" + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]
    
    out_treat = np.loadtxt('./data/IHDP/Series_y_' + "1" + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    # ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)
    ys = out_treat[:, 1:(t0 + 2)]

    x, X_test, y, y_test, t, t_test = train_test_split(matrix, ys, ts, test_size=0.2)
    
    model = MyModel_TransLTEE(t0,t)
    model_CF = MyModel_TransLTEE(t0,1-t)
    n_train = tf.cast(x.shape[0],tf.int32)
    # dim_x = tf.cast(x.shape[1],tf.int32)
    # dim_y = tf.cast(y.shape[1],tf.int32)
    print(x.shape, y.shape, t.shape, n_train)
    # _ = model.call(x,t,y[:,:-1],y[:,1:])
    # input_shape = [[n_train,dim_x],[n_train,dim_y-1],[n_train,dim_y-1]]
    # model.build(input_shape)
    # model.build(input_shape = input_shape)
    # model_CF.build(input_shape = input_shape)
    # model.get_weights()

    print('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION...')
    print('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')
    print('TRAINING MODEL...')

    ## Gradient Descent
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.lr, decay=config.weight_decay)
    sched = tf.keras.optimizers.schedules.ExponentialDecay(config.lr, decay_steps=3, decay_rate=0.5)
    loss_func = tf.keras.metrics.Mean(name='loss_func')
    accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_func')

    # predictions, predict_error, distance = model.call(
    #             x, t, y[:,:-1], y[:,1:])
    # print(predictions.shape, predict_error.shape, distance.shape)
    
    def train_step(X_train, tar_train, tar_real, gamma = 1e-4):

        with tf.GradientTape() as tape:
            predictions, predict_error, distance = model.call(
                X_train, tar_train, tar_real)
            
            CF_predictions, _, _ = model_CF(
                X_train, tar_train, tar_real)
            # model.build(input_shape=[N, TY.shape[1]])
            # loss_groundtruth = train_loss(abs(CF_predictions), groundtruth)
            # loss = predict_error + loss_groundtruth + gama*distance
            pred_effect = tf.reduce_mean(abs(predict_error-CF_predictions),axis=0)
            print(pred_effect)
            loss = predict_error + gamma * distance

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_func(loss)
        # train_accuracy(pred_effect, groundtruth)
        tar_real = tf.expand_dims(tar_real,-1)
        print(tar_real.shape, predictions.shape)
        accuracy_func(tar_real, predictions)
        # print([CF_predictions])

    ## Training
    for epoch in range(10):
        loss_func.reset_states()
        accuracy_func.reset_states()
        start = time.time()

        # tar_batch = tf.expand_dims(y_train[:,:-1],-1)
        # tar_real_batch = tf.expand_dims(y_train[:, 1:],-1)        

        # tf.keras.Model.compile(model,optimizer='Adam',
        #                        loss='sparse_categorical_crossentropy',
        #                        metrics=['accuracy'])
        # tf.keras.Model.fit(MyModel_TransLTEE,)
        train_step(x, y[:,:-1], y[:,1:])

        # Timing and printing happens here after each epoch
        epoch_time = time.time() - start
        print(f'Epoch {epoch + 1} Loss {loss_func.result():.4f} Accuracy {accuracy_func.result():.4f}')
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')

    print('MODEL SUCCESSFULLY TRAINED!')



if __name__ == '__main__':
    main()