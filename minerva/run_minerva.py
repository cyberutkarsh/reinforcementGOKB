from code.model.trainer import Trainer
from code.options import read_options
import logging
import sys
import json
import tensorflow as tf
import uuid
import os
import pprint

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Set logging
logger.setLevel(logging.INFO)

# Options
options = {}
# options['data_input_dir']="datasets/data_preprocessed/countries_S3/"
# options['vocab_dir']="datasets/data_preprocessed/countries_S3/vocab"
options['data_input_dir']="datasets/data_preprocessed/gokb/"
options['vocab_dir']="datasets/data_preprocessed/gokb/vocab"
options['total_iterations']=1000
options['path_length']=3
options['hidden_size']=50
options['embedding_size']=50
options['batch_size']=256
options['beta']=0.02
options['Lambda']=0.05
options['use_entity_embeddings']=0
options['train_entity_embeddings']=0
options['train_relation_embeddings']=1
# options['base_output_dir']="output/countries_s3/"
options['base_output_dir']="output/gokb/"
#options['model_load_dir']="output/gokb/4b39_3_0.01_10_0.0/model/model.ckpt"
options['model_load_dir']=""
options['load_model']=0
options['nell_evaluation']=0
options['num_rollouts']=20
options['test_rollouts']=10
options['LSTM_layers']=1
options['positive_reward']=1.0
options['negative_reward']=0.0
options['max_num_actions']=200
options['learning_rate']=1e-3
options['l2_reg_const']=1e-2
options['grad_clip_norm']=5
options['pretrained_embeddings_action']=""
options['pretrained_embeddings_entity']=""
options['gamma']=1
options['beta']=1e-2
options['Lambda']=0.02
options['pool']="max"
options['eval_every']=500
# options['output_dir'] = '' + 'output/countries_s3/' + str(uuid.uuid4())[:4]+'_'+str(options['path_length'])+'_'+str(options['beta'])+'_'+str(options['test_rollouts'])+'_'+str(options['Lambda'])
options['output_dir'] = '' + 'output/gokb/' + str(uuid.uuid4())[:4]+'_'+str(options['path_length'])+'_'+str(options['beta'])+'_'+str(options['test_rollouts'])+'_'+str(options['Lambda'])
options['model_dir'] = options['output_dir']+'/'+ 'model/'
 ##Logger##
options['path_logger_file'] = options['output_dir']
options['log_file_name'] = options['output_dir'] +'/log.txt'
os.makedirs(options['output_dir'])
os.mkdir(options['model_dir'])

fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
logfile = logging.FileHandler('reward.txt', 'w')
logfile.setFormatter(fmt)
logger.addHandler(logfile)
# read the vocab files, it will be used by many classes hence global scope
logger.info('reading vocab files...')
options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
logger.info('Reading mid to name map')
mid_to_word = {}
# with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
#     mid_to_word = json.load(f)
logger.info('Done..')
logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
save_path = ''
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.log_device_placement = False


#Training
if not options['load_model']:
    trainer = Trainer(options)
    with tf.Session(config=config) as sess:
        sess.run(trainer.initialize())
        trainer.initialize_pretrained_embeddings(sess=sess)

        trainer.train(sess)
        save_path = trainer.save_path
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

    tf.reset_default_graph()
#Testing on test with best model
else:
    logger.info("Skipping training")
    logger.info("Loading model from {}".format(options["model_load_dir"]))

trainer = Trainer(options)
if options['load_model']:
    save_path = options['model_load_dir']
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir
with tf.Session(config=config) as sess:
    trainer.initialize(restore=save_path, sess=sess)

    trainer.test_rollouts = 100

    os.mkdir(path_logger_file + "/" + "test_beam")
    trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    with open(output_dir + '/scores.txt', 'a') as score_file:
        score_file.write("Test (beam) scores with best model from " + save_path + "\n")
    trainer.test_environment = trainer.test_test_environment
    trainer.test_environment.test_rollouts = 100

    trainer.test(sess, beam=True, print_paths=True, save_model=False)


    print options['nell_evaluation']
    if options['nell_evaluation'] == 1:
        nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir+'/sort_test.pairs' )
