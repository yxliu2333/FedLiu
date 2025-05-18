import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import copy
from tensorflow.keras import layers, Model, losses, optimizers

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer


def pearson_corr(x, y, axis):
    """
    计算 x, y 在指定 axis 上的 Pearson 相关系数：
      - x, y: 张量，shape=(..., D, ...)
      - axis:  求相关的维度（1 表示对每个样本的 D 类别维度相关；0 表示对每个类别的样本维度相关）
    返回：沿 axis 后的相关值张量。
    """
    x_cent = x - tf.reduce_mean(x, axis=axis, keepdims=True)
    y_cent = y - tf.reduce_mean(y, axis=axis, keepdims=True)
    num = tf.reduce_sum(x_cent * y_cent, axis=axis)
    den = tf.norm(x_cent, axis=axis) * tf.norm(y_cent, axis=axis) + 1e-8
    return num / den

def inter_intra_loss(beta=1.0, gamma=1.0, tau=1.0):
    """
    返回只计算 L_inter 和 L_intra的复合损失函数，用于 logits 分支：
      输入 y_true_logits, y_pred_logits 均 shape=(batch_size, num_classes)
    L_inter: 样本级相关；L_intra: 类别级相关
    """
    def loss_fn(y_true_logits, y_pred_logits):
        # 温度缩放 logits 后得到概率
        p_s = tf.nn.softmax(y_pred_logits / tau, axis=1)  # shape (B, C)
        p_t = tf.nn.softmax(y_true_logits / tau, axis=1)  # shape (B, C)

        # 类间：对每个样本，在类别维度(1)上计算 Pearson，然后取平均
        corr_inter = pearson_corr(p_s, p_t, axis=1)        # shape (B,)
        L_inter = tau**2 * (1.0 - tf.reduce_mean(corr_inter))

        # 类内：对每个类别，在样本维度(0)上计算 Pearson，然后取平均
        corr_intra = pearson_corr(p_s, p_t, axis=0)        # shape (C,)
        L_intra = tau**2 * (1.0 - tf.reduce_mean(corr_intra))

        return beta * L_inter + gamma * L_intra
    return loss_fn

def kl_only_loss(tau=1.0):
    """
    仅基于聚合 logits (y_true) 与客户端 logits (y_pred) 的 KL 散度。
    y_true: 教师平均 logits, shape=(B, C)
    y_pred: 学生输出 logits,   shape=(B, C)
    """
    # 使用 TensorFlow 自带的 KLDivergence
    kld = tf.keras.losses.KLDivergence()
    def loss_fn(y_true, y_pred):
        # 对 logits 进行温度缩放并 softmax
        p_t = tf.nn.softmax(y_true / tau, axis=1)  # 教师概率
        p_s = tf.nn.softmax(y_pred / tau, axis=1)  # 学生概率
        # 计算 KL(p_t || p_s)，注意 KLDivergence 默认计算 KL(y_true ‖ y_pred)
        return kld(p_t, p_s) * (tau ** 2)
    return loss_fn

class FedMD():
    def __init__(self, parties, public_dataset, 
                 private_data, total_private_data,  
                 private_test_data, N_alignment,
                 N_rounds, 
                 N_logits_matching_round, logits_matching_batchsize, 
                 N_private_training_round, private_training_batchsize,beta,gamma,tauDIST,tauKL):
        
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment
        
        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        
        self.collaborative_parties = []
        self.collaborative_parties_copy = []
        self.init_result = []

        self.beta = beta
        self.gamma = gamma
        self.tauDIST = tauDIST
        self.tauKL = tauKL
        
        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                                 loss = "sparse_categorical_crossentropy",
                                 metrics = ["accuracy"])
            
            print("start full stack training ... ")        
            
            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                             validation_data = [private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                            )
            
            print("full stack training done")
            
            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")
            
            self.collaborative_parties.append({"model_logits": model_A, 
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            
            self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
                                     "train_acc": model_A_twin.history.history['accuracy'],
                                     "val_loss": model_A_twin.history.history['val_loss'],
                                     "train_loss": model_A_twin.history.history['loss'],
                                    })
            
            print()
            del model_A, model_A_twin
        #END FOR LOOP
        
        for i in range(self.N_parties):
            print("model ", i)
            model_A_copy_twin = None
            model_A_copy_twin = clone_model(parties[i])
            model_A_copy_twin.set_weights(parties[i].get_weights())
            model_A_copy_twin.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                                 loss = "sparse_categorical_crossentropy",
                                 metrics = ["accuracy"])
            
            print("2ndstart full stack training ... ")        
            
            model_A_copy_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                             validation_data = [private_test_data["X"], private_test_data["y"]],
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                            )
            
            print("2ndfull stack training done")
            
            model_A_copy = remove_last_layer(model_A_copy_twin, loss="mean_absolute_error")
            
            self.collaborative_parties_copy.append({"model_logits": model_A_copy, 
                                               "model_classifier": model_A_copy_twin,
                                               "model_weights": model_A_copy_twin.get_weights()})
            
            print()
            del model_A_copy, model_A_copy_twin
        #END FOR LOOP

        print("calculate the theoretical upper bounds for participants: ")
        
        self.upper_bounds = []
        self.pooled_train_result = []
        for model in parties:
            model_ub = clone_model(model)
            model_ub.set_weights(model.get_weights())
            model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
                             loss = "sparse_categorical_crossentropy", 
                             metrics = ["accuracy"])
            
            model_ub.fit(total_private_data["X"], total_private_data["y"],
                         batch_size = 32, epochs = 50, shuffle=True, verbose = 0, 
                         validation_data = [private_test_data["X"], private_test_data["y"]],
                         callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10)])
            
            self.upper_bounds.append(model_ub.history.history["val_accuracy"][-1])
            self.pooled_train_result.append({"val_acc": model_ub.history.history["val_accuracy"], 
                                             "acc": model_ub.history.history["accuracy"]})
            
            del model_ub    
        print("the upper bounds are:", self.upper_bounds)
    
    def collaborative_training(self):
        # start collaborating training    
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        collaboration_performance_copy = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"], 
                                                     self.public_dataset["y"],
                                                     self.N_alignment)
            
            print("round ", r)
            
            print("update logits ... ")
            # update logits
            logits = 0
            logits_copy = 0
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                logits += d["model_logits"].predict(alignment_data["X"], verbose = 0)
                
            logits /= self.N_parties
            
            for d in self.collaborative_parties_copy:
                d["model_logits"].set_weights(d["model_weights"])
                logits_copy += d["model_logits"].predict(alignment_data["X"], verbose = 0)
                
            logits_copy /= self.N_parties

            # test performance
            print("test performance ... ")

            # 初始化浮点数数组用于装结果
            tensor_a = tf.zeros(shape=(0,), dtype=tf.float32)
            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                print("dist:")
                print(collaboration_performance[index][-1])
                normalized_value = collaboration_performance[index][-1] / self.upper_bounds[index]
                tensor_a = tf.concat([tensor_a, [normalized_value]], axis=0)
                del y_pred
            print("tensor_a 的值：", tensor_a.numpy())
            mean_value = tf.reduce_mean(tensor_a)
            print("平均值：", mean_value.numpy())
            variance_value = tf.math.reduce_variance(tensor_a)
            print("方差：", variance_value.numpy())

            # 初始化浮点数数组用于装结果
            tensor_b = tf.zeros(shape=(0,), dtype=tf.float32)
            for index, d in enumerate(self.collaborative_parties_copy):
                y_pred_copy = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                collaboration_performance_copy[index].append(np.mean(self.private_test_data["y"] == y_pred_copy))
                print("kl:")
                print(collaboration_performance_copy[index][-1])
                normalized_value_copy = collaboration_performance_copy[index][-1] / self.upper_bounds[index]
                tensor_b = tf.concat([tensor_b, [normalized_value_copy]], axis=0)
                del y_pred_copy
            print("tensor_b 的值：", tensor_b.numpy())
            mean_value_copy = tf.reduce_mean(tensor_b)
            print("平均值：", mean_value_copy.numpy())
            variance_value_copy = tf.math.reduce_variance(tensor_b)
            print("方差：", variance_value_copy.numpy())
                
            r+= 1
            if r > self.N_rounds:
                break
                
                
            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss=inter_intra_loss(beta=self.beta, gamma=self.gamma, tau=self.tauDIST)
                )   
                model_B.fit(
                    alignment_data["X"], logits,
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                print("dist_model {0} done alignment".format(index))

                print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            for index, d in enumerate(self.collaborative_parties_copy):
                print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss=kl_only_loss(tau=self.tauKL)
                )   
                model_C.fit(
                    alignment_data["X"], logits_copy,
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w_copy = model_C.get_weights()
                d["model_weights"] = new_w_copy
                d["model_logits"].set_weights(new_w_copy)
                
                print("kl_model {0} done alignment".format(index))

                print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP
        
        #END WHILE LOOP
        return collaboration_performance


        