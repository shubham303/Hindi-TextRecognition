import torch
import torch.nn.functional as F
import tqdm
from torchvision import transforms

import converters.Attn_Indic_converter
import loggers
import options.train_options
from dataset.combined_dataset import ConcatDatasets
from dataset.lmdb_dataset import LmdbDataset
from loss.cross_entropy_loss import CrossEntropyLoss
from metrics.accuracy import Accuracy
from models.model import STRHindi
import wandb


def load_criterion(criterion, ignore_index):
    if criterion == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)


def load_optimizer(optim, lr, weight_decay, model):
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer {}".format(optim))


def load_lr_scheduler(lr_scheduler, optimizer):
    if lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, verbose=True)


def load_model(model_name, num_characters):
    return STRHindi(num_characters=num_characters)


def postprocess(preds, converter):
    probs = F.softmax(preds, dim=2)
    max_probs, indexes = probs.max(dim=2)

    preds_str = []
    preds_prob = []
    for i, pstr in enumerate(converter.decode(indexes)):
        str_len = len(pstr)
        if str_len == 0:
            prob = 0
        else:
            prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
        preds_prob.append(prob)
        preds_str.append(pstr)

    return preds_str, preds_prob


def train():
    args_parser = options.train_options.get_train_options()
    args = args_parser.parse_args()
    print(args)
    device = torch.device("cuda:0")
    converter = converters.converter.AttnIndicconverter(args.character, args.batch_max_length)
    model = load_model(args.model_name, num_characters=len(converter.character)).to(device)
    model.train()
    criterion = load_criterion(args.criterion, converter.ignore_index)

    optimizer = load_optimizer(args.optim, args.lr, args.weight_decay, model)

    scheduler = load_lr_scheduler("cosine", optimizer)

    logger = loggers.get_logger()  # todo pass arguments

    # TODO define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    )

    datasets = []
    if args.mj_root != "":
        # number of characters is batch_max_len
        datasets.append(
            LmdbDataset(args.mj_root, character=args.character, batch_max_length=args.word_len, transform=transform,
                        max_tokens=args.batch_max_length // args.char_per_token))

    if args.st_root != "":
        datasets.append(
            LmdbDataset(args.st_root, character=args.character, batch_max_length=args.word_len, transform=transform,
                        max_tokens=args.batch_max_length // args.char_per_token))

    dataset = ConcatDatasets(datasets, [0.5, 0.5])
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    metric = Accuracy()
    for epoch in range(args.epochs):
        for iter, data in tqdm.tqdm(enumerate(dataloader)):
            images, labels = data
            images = images.to(device)
            optimizer.zero_grad()

            label_input, label_len, label_target, lang_id = converter.train_encode(labels)

            label_input, label_len, label_target, lang_id = label_input.to(device), label_len.to(
                device), label_target.to(device), lang_id.to(device)

            out, memory, text_embedding = model(images, label_input)

            loss = criterion(out, label_target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred, prob = postprocess(out, converter)
                metric.measure(pred, prob, labels)

            if iter != 0 and iter % 100 == 0:
                logger.info(
                    'Train, Epoch %d, Iter %d, LR %s, Loss %.4f, '
                    'acc %.4f, edit_distance %s'
                    % (epoch, iter, optimizer.param_groups[0]['lr'], loss.item(),
                       metric.avg['acc']['true'], metric.avg['edit']))

                logger.info(f'\n{metric.predict_example_log}')
                if iter % 1000 == 0:
                    torch.save(model.state_dict(), "checkpoints/model_{}_{}.pth".format(iter, epoch))

        scheduler.step()


wandb.init(name="first experiment")
train()
wandb.finish()
