import tensorflow as tf

def build_densenet121(
    input_shape,
    output_dim,
    is_train_feature_extractor_layer=True):
    """
    transfer learning using imagenet weights
    """
    feature_extractor_layer = \
        tf.keras.applications.DenseNet121(
            include_top=False, 
            input_shape=input_shape, 
            weights='imagenet', 
            pooling='avg') 
    feature_extractor_layer.trainable = is_train_feature_extractor_layer

    model = tf.keras.Sequential([
        feature_extractor_layer,
        # If you are using CIFAR100, you will need to reduce the layer size.
        tf.keras.layers.Dense(128,        activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model

def build_nasnet(
    input_shape,
    output_dim,
    is_train_feature_extractor_layer=True):
    """
    transfer learning using imagenet weights
    """
    feature_extractor_layer = \
        tf.keras.applications.nasnet.NASNetMobile(
            include_top=False,
            input_shape=input_shape, 
            weights='imagenet', 
            pooling='avg') 
    feature_extractor_layer.trainable = is_train_feature_extractor_layer

    model = tf.keras.Sequential([
        feature_extractor_layer,
        # If you are using CIFAR100, you will need to reduce the layer size.
        tf.keras.layers.Dense(128,        activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model

def build_mobilenet(
    input_shape,
    output_dim,
    is_train_feature_extractor_layer=True):
    """
    transfer learning using imagenet weights
    """
    feature_extractor_layer = \
    tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=input_shape, 
        weights='imagenet', 
        pooling='avg')
    feature_extractor_layer.trainable = is_train_feature_extractor_layer

    model = tf.keras.Sequential([
        feature_extractor_layer,
        # If you are using CIFAR100, you will need to reduce the layer size.
        tf.keras.layers.Dense(128,        activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model


def build_resnet50(
    input_shape,
    output_dim,
    is_train_feature_extractor_layer=True):
    """
    transfer learning using imagenet weights
    """
    feature_extractor_layer = \
    tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_shape=input_shape, 
        weights='imagenet', 
        pooling='avg')
    feature_extractor_layer.trainable = is_train_feature_extractor_layer

    model = tf.keras.Sequential([
        feature_extractor_layer,
        # If you are using CIFAR100, you will need to reduce the layer size.
        tf.keras.layers.Dense(128,        activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model
