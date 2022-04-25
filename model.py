def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=1, padding="same")(x) #(6 * rate_scale, 6 * rate_scale)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x2 = Conv2D(num_filters, (3, 3), dilation_rate=2, padding="same")(x) #(12 * rate_scale, 12 * rate_scale)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    
    x3 = Conv2D(num_filters, (3, 3), dilation_rate=4, padding="same")(x) #(18 * rate_scale, 18 * rate_scale)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    
    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    return y

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = Dropout(0.5)(x)
    x = conv_block(x, num_filters)
    return x

def build_densenet201_ASPP_unet(input_shape, num_classes):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained DenseNet121 Model """
    densenet = DenseNet201(include_top=False, weights="imagenet", input_tensor=inputs)
    #densenet.summary()
    densenet.trainable = False
    #####Depth 4####
    """ Encoder """
    s1 = densenet.get_layer("input_1").output       ## 512
    s2 = densenet.get_layer("conv1/relu").output    ## 256
    s3 = densenet.get_layer("pool2_relu").output    ## 128
    s4 = densenet.get_layer("pool3_relu").output    ## 64

    """ Bridge """
    b1 = densenet.get_layer("pool4_relu").output    ## 32
    b2 = aspp_block(b1, 1792)
    b2 = Dropout(0.5)(b2)

    """ Decoder """
    d1 = decoder_block(b2, s4, 512)             ## 64
    d2 = decoder_block(d1, s3, 256)             ## 128
    d3 = decoder_block(d2, s2, 128)             ## 256
    d4 = decoder_block(d3, s1, 64)              ## 512
    """ Outputs """
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs)
    return model