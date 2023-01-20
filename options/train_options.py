import argparse


def get_train_options():
    arg_parser = argparse.ArgumentParser(description='Experiment setup')

    arg_parser.add_argument("--character", type=str,
                            default=')+*/,.:;=>]-_|~%}{([कखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहक़ख़ग़ज़ड़ढ़फ़य़ॸॹॺॻॼॽॾॿऄअआइईउऊऋऌऍऎएऐऑऒओऔॠॡॲॳॴॵॶॷऺऻािीुूृॄॅॆेैॉॊोौॎॏॕॖॗॢॣऀँंःऽ़्०१२३४५६७८९०१२३४५६७८९)+*/,.:;=>]-_|~%}{([।॥ॐ')
    arg_parser.add_argument("--test_character", type=str,
                            default=')+*/,.:;=>]-_|~%}{([कखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहक़ख़ग़ज़ड़ढ़फ़य़ॸॹॺॻॼॽॾॿऄअआइईउऊऋऌऍऎएऐऑऒओऔॠॡॲॳॴॵॶॷऺऻािीुूृॄॅॆेैॉॊोौॎॏॕॖॗॢॣऀँंःऽ़्०१२३४५६७८९०१२३४५६७८९)+*/,.:;=>]-_|~%}{([।॥ॐ')
    arg_parser.add_argument("--criterion", type=str, default="ce", help="cross entropy loss")
    arg_parser.add_argument("--batch_max_length", type=int, default=70)  # number of characters
    arg_parser.add_argument("--char_per_token", type=int,
                            default=7)  # numeber of characters per token , so max_tokens = 70/7 =10
    arg_parser.add_argument("--optim", type=str, default="adam")
    arg_parser.add_argument("--lr", type=float, default="0.01")
    arg_parser.add_argument("--weight_decay", type=float, default="0.0")

    arg_parser.add_argument("--st_root", type=str,
                            default="/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi/training/ST/")
    arg_parser.add_argument("--mj_root", type=str,
                            default="/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi/training/MJ/MJ_train")

    arg_parser.add_argument("--word_len", type=int, default=35)  # number of characters in word
    arg_parser.add_argument("--model_name", type=str, default="")
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--exp_name", type=str, default="")
    arg_parser.add_argument("--epochs", type=int, default=10)
    return arg_parser
