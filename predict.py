import argparse
from util_functions import input_args, load_json, checkpoint, predict, plot_predict

#filepath = test_dir + '/102/image_08004.jpg'     #blackberry lily
#filepath = test_dir + '/66/image_05549.jpg'      #osteospermum
#filepath = test_dir + '/16/image_06670.jpg'      #globe-flower
#filepath = test_dir + '/1/image_06760.jpg'       #pink primrose (get pelargonium instead)
#filepath = test_dir + '/11/image_03098.jpg'      #snapdragon
#filepath = 'flowers/test/1/image_06764.jpg'      #pink primrose

#get inputs
arg = input_args()
filepath = arg.img_path             #test image for prediction
cat_to_name= load_json(arg.cat_names)

# Find prediction
if arg.show == True:
    #show plot
    plot_predict(filepath, arg.ckpt_dir)
else:
    #show top few probs and classes
    probs, classes = predict(filepath, arg.ckpt_dir, arg.top_k)
    
    flower_names = [cat_to_name[k] for k in classes]                        #get names based on class number
    
    print('\nPredicting on: ', filepath)                                    #print predicting image path
    print('Top {} Probabilities: {}'.format(arg.top_k, probs))
    print('Top {} Classes: {}'.format(arg.top_k, classes))
    print('Top {} Flower Names: {}'.format(arg.top_k, flower_names))