from preproc import *
from utils import *

import argparse
import logging
import sagemaker_containers
import time
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
def _train(args):
    t_total = 268 // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(t_total * 0.1)

    # args.data_dir ==  /opt/ml/input/data/training
    logger.info("Loading Data")
    tokenizer = define_tokenizer()
    X_train, mask_train, y_train = prepare_dataset(
        args.data_dir + '/in_domain_train.tsv', tokenizer, args.max_len)
    X_valid, mask_valid, y_valid = prepare_dataset(
        args.data_dir + '/in_domain_dev.tsv', tokenizer, args.max_len)

    train_dataloader = transer_gpu_dataloader(X_train, mask_train, y_train, args.batch_size, device)
    valid_dataloader = transer_gpu_dataloader(X_valid, mask_valid, y_valid, args.batch_size, device)

    logger.info("Model loaded")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # if torch.cuda.device_count() > 1:
    #    logger.info("Gpu count: {}".format(torch.cuda.device_count()))
    #model = torch.nn.DataParallel(model)

    model = model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    train_start = time.time()

    train_loss_set = []
    learning_rate = []

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        model.train()
        tr_loss, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            learning_rate.append(scheduler.get_lr())
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            tr_loss += loss.item()
            nb_tr_steps += 1
        print('Train loss: {}'.format(tr_loss/nb_tr_steps))

        model.eval()
        predictions = []
        true_labels = []

        for batch in valid_dataloader:
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            predictions.append(np.argmax(logits[0].detach().cpu().numpy(), axis=1))
            true_labels.append(b_labels.to('cpu').numpy())

    print('Training Time (m)', (time.time() - train_start)/60)
    logger.info("Training Completed!")

    logger.info("Evaluation Step")
    model.eval()
    predictions = []
    true_labels = []
    for batch in valid_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    predictions.append(np.argmax(logits[0].detach().cpu().numpy(), axis=1))
    true_labels.append(b_labels.to('cpu').numpy())

    predictions = flatten_list(predictions)
    true_labels = flatten_list(true_labels)

    print('Validation Accuracy: {}'.format(simple_accuracy(predictions, true_labels)))
    print('Matthew Correlation: {}'.format(matthew_correlation(predictions, true_labels)))

    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    # if torch.cuda.device_count() > 1:
    #    logger.info("Gpu count: {}".format(torch.cuda.device_count()))
    #    model = torch.nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

# HYPERPARAMS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64, metavar='LEN',
                        help='max length (default: 64)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='GAS',
                        help='gradient_accumulation_steps (default: 1)')
    parser.add_argument('--num_train_epochs', type=int, default=4, metavar='EPOCH',
                        help='num_train_epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='initial learning rate (default: 5e-5)')
    parser.add_argument('--eps', type=float, default=1, metavar='EPS', help='eps (default: 1e-8)')
    parser.add_argument('--weight_decay', type=float, metavar='DECAY',
                        default=0.0, help='weight decay (default: 0.0)')

    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _train(parser.parse_args())
