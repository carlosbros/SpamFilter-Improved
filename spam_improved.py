############################################################
# Spam Filter Improved
############################################################

import email
import math
import os

def load_tokens(email_path):
    with open(email_path) as email_file:
        message = email.message_from_file(email_file)
    return [token for line in email.iterators.body_line_iterator(message)
            for token in line.split()]


def get_bigram_tokens(tokens):
    return [" ".join([tokens[i - 1], tokens[i]])
            for i in xrange(1, len(tokens))]


def log_probs(email_paths, smoothing_u, smoothing_b):
    count_dict = {}
    log_prob_dict = {}
    total_count = 0
    num_words = 0

    count_dict["<< len<7 >>"] = 0
    count_dict["<< 7<len<13 >>"] = 0
    count_dict["<< len>13 >>"] = 0
    count_dict["<< digit_count >>"] = 0
    for email_path in email_paths:
        tokens = load_tokens(email_path)
        tokens.extend(get_bigram_tokens(tokens))
        for token in tokens:
            count_dict[token] = count_dict.get(token, 0) + 1

            # Take into account length of unigram tokens
            if len(token.split()) == 1:
                if len(token) < 7:
                    count_dict["<< len<7 >>"] += 1
                elif len(token) >= 7 and len(token) < 13:
                    count_dict["<< 7<len<13 >>"] += 1
                elif len(token) > 13:
                    count_dict["<< len>13 >>"] += 1

            # Take into account whether the token is a digit
            if token.isdigit():
                count_dict["<< digit_count >>"] += 1

    count_keys = count_dict.keys()
    for key in count_keys:
        if len(key.split()) == 2 and count_dict[key] <= 1:
            del count_dict[key]

    for w in count_dict:
        total_count += count_dict[w]
        num_words += 1

    # print "l<7: " + str(count_dict["<< len<7 >>"])
    # print "7<l<13: " + str(count_dict["<< 7<len<13 >>"])
    # print "l>13: " + str(count_dict["<< len>13 >>"])
    # print "num_words: " + str(total_count)

    # Compute log probabilities
    for token in count_dict.keys():
        smoothing = smoothing_u
        if len(token.split()) == 2:
            smoothing = smoothing_b
        numerator = count_dict[token] + smoothing
        denominator = total_count + (smoothing * (num_words + 1))
        log_prob_dict[token] = math.log(numerator / denominator)

    log_prob_dict["<UNK>"] = math.log(smoothing_u / denominator)
    return log_prob_dict


class SpamFilter(object):

    SMOOTH = 1e-9
    SMOOTH_B = 1e-15

    def __init__(self, spam_dir, ham_dir):
        spam_filenames = [spam_dir + "/" + f for f in os.listdir(spam_dir)]
        ham_filenames = [ham_dir + "/" + f for f in os.listdir(ham_dir)]
        self.spam_dict = log_probs(spam_filenames, self.SMOOTH, self.SMOOTH_B)
        self.ham_dict = log_probs(ham_filenames, self.SMOOTH, self.SMOOTH_B)
        num_spam_files = float(len(spam_filenames))
        num_ham_files = float(len(ham_filenames))
        num_files = num_spam_files + num_ham_files
        self.log_p_spam = math.log(num_spam_files / num_files)
        self.log_p_ham = math.log(num_ham_files / num_files)

    def is_spam(self, email_path):
        tokens = load_tokens(email_path)
        tokens.extend(get_bigram_tokens(tokens))
        spam_sum = 0
        ham_sum = 0

        for token in tokens:
            if token in self.spam_dict:
                spam_sum += self.spam_dict[token]
            else:
                spam_sum += self.spam_dict["<UNK>"]
            if token in self.ham_dict:
                ham_sum += self.ham_dict[token]
            else:
                ham_sum += self.ham_dict["<UNK>"]

            # Take into account the length of unigram tokens
            if len(token.split()) == 1:
                if len(token) < 7:
                    spam_sum += self.spam_dict["<< len<7 >>"]
                    ham_sum += self.ham_dict["<< len<7 >>"]
                elif len(token) >= 7 and len(token) < 13:
                    spam_sum += self.spam_dict["<< 7<len<13 >>"]
                    ham_sum += self.ham_dict["<< 7<len<13 >>"]
                elif len(token) > 13:
                    spam_sum += self.spam_dict["<< len>13 >>"]
                    ham_sum += self.ham_dict["<< len>13 >>"]

            # Take into account whether the token is a digit
            if token.isdigit():
                spam_sum += self.spam_dict["<< digit_count >>"]
                ham_sum += self.ham_dict["<< digit_count >>"]

        spam_sum += self.log_p_spam
        ham_sum += self.log_p_ham
        return spam_sum > ham_sum


# ==========================
# ---- ACCURACY TESTING ----
# ==========================
# spam_dir = "data/train/spam"
# ham_dir = "data/train/ham"
# sf = SpamFilter(spam_dir, ham_dir)
# spam_dir = "data/dev/spam"
# ham_dir = "data/dev/ham"
# spam_filenames = [spam_dir + "/" + f for f in os.listdir(spam_dir)]
# ham_filenames = [ham_dir + "/" + f for f in os.listdir(ham_dir)]
# count = 0
# counter = 0
# count_h = 0
# counter_h = 0
# for e_path in spam_filenames:
#     counter += 1
#     if sf.is_spam(e_path):
#         count += 1
#     else:
#         print e_path
# for e_path in ham_filenames:
#     counter_h += 1
#     if not sf.is_spam(e_path):
#         count_h += 1
#     else:
#         print e_path
# print "Spam accuracy: " + str((float(count) / counter) * 100) # It's 99.5%
# print "Ham accuracy: " + str((float(count_h) / counter_h) * 100) # It's 99.5%
