# Request_LLM() = Request for access to your LLM
# Due to privacy factors, this code is only used as a functional reference and cannot be run directly



class Reranking():
    def __init__(self, dataset_name, nodes=["diversity", "accuracy", "stop"],
                 user_fea=None, his_item_fea=None, item_fea=None, top_k=10, re_history=[[], []],
                 data=None, focus="Overall Performance", history_max=5, max_count=3):
        self.nodes = nodes
        self.user_fea = user_fea
        self.his_item_fea = his_item_fea
        self.item_fea = item_fea
        self.top_k = top_k
        self.re_history = re_history
        self.data = data
        self.candidate_id = None
        self.focus = focus
        self.history_max = history_max
        self.max_count = max_count
        self.dataset_name = dataset_name

    def update_data(self, data):
        self.data = data
        self.candidate_id = self.data["candidate_items"]["item_id"].tolist()

    def request(self, prompt, current_name):
        print("Requesting...")
        response_ori = Request_LLM(
            model="your LLM",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content
        while True:
            try:
                if current_name == 'backward':
                    print("Backwarding and redirecting...")
                    return getattr(self, response_ori + '_rerank')()

                flag = 0
                while True:
                    response = [[i for i in response_ori.split("[")[-1].split("]")[0].split(", ")],
                                response_ori.split(";")[-1].strip()]
                    response[1] = response[1].lower().strip("_,. ").replace('"', '').replace("'", "")
                    # check id
                    for re in response[0]:
                        if int(re) not in self.candidate_id:
                            flag += 1
                            response_ori = Request_LLM(
                                model="your LLM",
                                messages=[
                                    {"role": "user", "content": prompt},
                                    {"role": "system", "content": response_ori},
                                    {"role": "user",
                                     "content": "Reranking id '{}' in your reply does not appear in the candidates provided. "
                                                "You should make sure that your reranking list is a reorder of the original "
                                                "candidate ids, including each candidate id once and only once. "
                                                "Please check and answer again.".format(re)},
                                ]
                            ).choices[0].message.content
                            print("Response incorrect, answer id not in candidates: ", response[0])
                            break
                    for re in self.candidate_id:
                        if str(re) not in response[0]:
                            flag += 1
                            response_ori = Request_LLM(
                                model="your LLM",
                                messages=[
                                    {"role": "user", "content": prompt},
                                    {"role": "system", "content": response_ori},
                                    {"role": "user",
                                     "content": "Candidate item id '{}' does not appear in the reranking list of your reply\item"
                                                "You should make sure that your reranking list is a reorder of the original "
                                                "candidate ids, including each candidate id once and only once. "
                                                "Please check and answer again.".format(str(re))},
                                ]
                            ).choices[0].message.content
                            print("Response incorrect, candidate id not in answer: ", response[0])
                            break
                    if flag > 0:
                        flag = 0
                        continue
                    # check id num
                    if len(response[0]) != len(self.candidate_id):
                        response_ori = Request_LLM(
                            model="your LLM",
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "system", "content": response_ori},
                                {"role": "user",
                                 "content": "The number of ids in the reply does not match the required number ({}). "
                                            "You should make sure that your reranking list is a reorder of the original "
                                            "candidate ids, including each candidate id once and only once. "
                                            "Please check and answer again.".format(str(len(self.candidate_id)))},
                            ]
                        ).choices[0].message.content
                        print("Response incorrect, reranking length error: ", response[0])
                        continue
                    # check node name
                    if response[1] not in self.nodes:
                        response_ori = Request_LLM(
                            model="your LLM",
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "system", "content": response_ori},
                                {"role": "user",
                                 "content": "The node name doesn't exist in ({}). "
                                            "You should make sure that a correct node name is followed by your reranking list. "
                                            "Please check and answer again.".format(','.join(self.nodes))},
                            ]
                        ).choices[0].message.content
                        print("Response incorrect, node name error: ", response[1])
                        continue
                    # response correct
                    print("Response correct: ", response[0])
                    break
            except:
                response_ori = Request_LLM(
                    model="your LLM",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": response_ori},
                        {"role": "user",
                         "content": "Answer format or is incorrect. Please check and answer again."},
                    ]
                ).choices[0].message.content
                print("Response incorrect, reranking format error: ", response)
                continue
            else:
                self.re_history[0].append(current_name)
                self.re_history[1].append(response[0])
                if len(self.re_history[0]) >= self.max_count:
                    response[1] = 'stop'
                return getattr(self, response[1] + '_rerank')()

    def backward_rerank(self):
        current_name = 'backward'
        self.re_history[0] = self.re_history[0][:-1]
        self.re_history[1] = self.re_history[1][:-1]

        prompt = "Considering a user, his/her basic infomation is: \n{"
        for fea in self.user_fea:
            prompt += "{}:{},".format(fea, str(self.data["user_info"][fea]))
        prompt += "}\nHis/Her history of browsing items and related features are: \n{"
        his_count = 1
        for _, item in self.data["history_items"].iterrows():
            prompt += "["
            for fea in self.his_item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += "]\n"
            his_count += 1
            if his_count > self.history_max:
                break
        prompt += "}\nHere's a list of the candidate items (with related features) he/she might see next: \n{"
        for _, item in self.data["candidate_items"].iterrows():
            prompt += "["
            for fea in self.item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += ("]\n")
        prompt += ("}\nNote that the 'score' feature is generated by an existing"
                   " recommendation model for your reference. ")

        if len(self.re_history[0]) > 0:
            prompt += "Your historical operations and reranking results represented by item_id are as follows: {\n"
            for i in range(len(self.re_history[0])):
                prompt += "Focus: {}. Reranking Result: [{}]\n".format(self.re_history[0][i],
                                                                       ", ".join(self.re_history[1][i]))
            prompt += "}\n"
        else:
            prompt += ("Your historical operations and reranking results represented by item_id are as follows: \n"
                       "{No historical operations yet.}\n")
        if len(self.re_history[0]) > 0:
            prompt += (
                    "Now, you need to give suggestions about the next step of reranking from the following reranking "
                    "focus: \n{" + ', '.join(
                self.nodes) + "}\n"
                              "Specially, 'stop' means to stop reranking in the next step "
                              "and output the result of the current step as the final reranking output. And 'backward' means to "
                              "delete the latest reranking operation and result so that they are not taken into consideration by"
                              " subsequent operations.\n"
                              "Your decision should be based on your final goal of the reranking:\n{" + self.focus + "}.\n")
        else:
            nodes = self.nodes.copy()
            nodes.remove('stop')
            prompt += (
                    "Now, you need to give suggestions about the next step of reranking from the following reranking "
                    "focus: \n{" + ', '.join(
                nodes) + "}\nSpecially, 'backward' means to "
                         "delete the latest reranking operation and result so that they are not taken into consideration by"
                         " subsequent operations.\n"
                         "Your decision should be based on your final goal of the reranking:\n{" + self.focus + "}.\n")

        prompt += (
            "\nFor your response format, please only give me a word of operation name you suggest to do next from "
            "the list of reranking focus as your answer without any punctuation,"
            " and omit anything else such as your thinking and decision-making process.")

        return self.request(prompt, current_name)

    def stop_rerank(self):
        current_name = 'stop'
        re_history = self.re_history
        self.re_history = [[], []]
        response = [int(i) for i in re_history[1][-1]]
        score = [response.index(i) for i in self.candidate_id]
        return response, re_history, score

    def accuracy_rerank(self):
        current_name = 'accuracy'
        prompt = "Considering a user, his/her basic infomation is: \n{"
        for fea in self.user_fea:
            prompt += "{}:{},".format(fea, str(self.data["user_info"][fea]))
        prompt += "}\nHis/Her history of browsing items and related features are: \n{"
        his_count = 1
        for _, item in self.data["history_items"].iterrows():
            prompt += "["
            for fea in self.his_item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += "]\n"
            his_count += 1
            if his_count > self.history_max:
                break
        prompt += "}\nHere's a list of the candidate items (with related features) he/she might see next: \n{"
        for _, item in self.data["candidate_items"].iterrows():
            prompt += "["
            for fea in self.item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += ("]\n")
        prompt += ("}\nNote that the 'score' feature is generated by an existing"
                   " recommendation model for your reference. ")

        if len(self.re_history[0]) > 0:
            prompt += "Your historical operations and reranking results represented by item_id are as follows: {\n"
            for i in range(len(self.re_history[0])):
                prompt += "Focus: {}. Reranking Result: [{}]\n".format(self.re_history[0][i],
                                                                       ", ".join(self.re_history[1][i]))
            prompt += "}\n"
        else:
            prompt += ("Your historical operations and reranking results represented by item_id are as follows: \n"
                       "{No historical operations yet.}\n")

        prompt += ("Now, you need to focus on the " + current_name + " aspect (the match between the user and items) "
                                                                     "and rerank the candidates based on the given "
                                                                     "information, and then give suggestions about the next step of reranking from the following reranking "
                                                                     "focus: \n{" + ', '.join(
            self.nodes) + "}\nSpecially, 'stop' means to stop reranking in the next step "
                          "and output the result of the current step as the final reranking output. And 'backward' means to "
                          "delete the latest reranking operation and result so that they are not taken into consideration by"
                          " subsequent operations.")

        prompt += (
                "\nThe order of reranking result should represent how likely the user is to watch it."
                "For suggestions about the next step, You should choose one of "
                "the functions representing the next reranking focus or 'stop' based on the final goal of the reranking:\n{" + self.focus + "}.\n"

                                                                                                                                            "For your response format, please only give me a list of item_id (containing all item_id values in the "
                                                                                                                                            "candidates list provided above and not containing any other item_id not mentioned) in order of recommendation priority and "
                                                                                                                                            "an operation name you suggest to do next from the list of reranking focus as your answer,"
                                                                                                                                            " and omit anything else such as your thinking and decision-making "
                                                                                                                                            "process. \nYour answer should strictly follow the format provided below containing a list of reranking result "
                                                                                                                                            "with all the item_id in the candidates and a word indicating the next operation."
                                                                                                                                            "\nExample answer format for 20 candidates: \n[16, 1246, 536, 15, 748, 1135, 1636, "
                                                                                                                                            "1, 47, 8, 2478, 899, 151, 1032, 165, 363, 527, 2168, 37, 62]; stop")

        return self.request(prompt, current_name)

    def diversity_rerank(self):
        current_name = 'diversity'
        prompt = "Considering a user, his/her basic infomation is: \n{"
        for fea in self.user_fea:
            prompt += "{}:{},".format(fea, str(self.data["user_info"][fea]))
        prompt += "}\nHis/Her history of browsing items and related features are: \n{"
        his_count = 1
        for _, item in self.data["history_items"].iterrows():
            prompt += "["
            for fea in self.his_item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += "]\n"
            his_count += 1
            if his_count > self.history_max:
                break
        prompt += "}\nHere's a list of the candidate items (with related features) he/she might see next: \n{"
        for _, item in self.data["candidate_items"].iterrows():
            prompt += "["
            for fea in self.item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += ("]\n")
        prompt += ("}\nNote that the 'score' feature is generated by an existing"
                   " recommendation model for your reference. ")

        if len(self.re_history[0]) > 0:
            prompt += "Your historical operations and reranking results represented by item_id are as follows: {\n"
            for i in range(len(self.re_history[0])):
                prompt += "Focus: {}. Reranking Result: [{}]\n".format(self.re_history[0][i],
                                                                       ", ".join(self.re_history[1][i]))
            prompt += "}\n"
        else:
            prompt += ("Your historical operations and reranking results represented by item_id are as follows: \n"
                       "{No historical operations yet.}\n")
        if self.dataset_name == 'ml-1m':
            prompt += (
                    "Now, you need to focus on the " + current_name + " aspect (more items with different 'genre' feature "
                                                                      "at the top of the list)")
        elif self.dataset_name == 'kuairand':
            prompt += (
                    "Now, you need to focus on the " + current_name + " aspect (more items with different 'upload_type' feature "
                                                                      "at the top of the list)")
        prompt += (" and rerank the candidates based on the given "
                   "information, and then give suggestions about the next step of reranking from the following reranking "
                   "focus: \n{" + ', '.join(
            self.nodes) + "}\nSpecially, 'stop' means to stop reranking in the next step "
                          "and output the result of the current step as the final reranking output. And 'backward' means to "
                          "delete the latest reranking operation and result so that they are not taken into consideration by"
                          " subsequent operations.")

        prompt += (
                "\nThe order of reranking result should represent how likely the user is to watch it."
                "For suggestions about the next step, You should choose one of "
                "the functions representing the next reranking focus or 'stop' based on the final goal of the reranking:\n{" + self.focus + "}.\n"

                                                                                                                                            "For your response format, please only give me a list of item_id (containing all item_id values in the "
                                                                                                                                            "candidates list provided above and not containing any other item_id not mentioned) in order of recommendation priority and "
                                                                                                                                            "an operation name you suggest to do next from the list of reranking focus as your answer,"
                                                                                                                                            " and omit anything else such as your thinking and decision-making "
                                                                                                                                            "process. \nYour answer should strictly follow the format provided below containing a list of reranking result "
                                                                                                                                            "with all the item_id in the candidates and a word indicating the next operation. Punctuations marks not mentioned "
                                                                                                                                            "in the example should not be included in your answer. "
                                                                                                                                            "\nExample answer format for 20 candidates: \n[16, 1246, 536, 15, 748, 1135, 1636, "
                                                                                                                                            "1, 47, 8, 2478, 899, 151, 1032, 165, 363, 527, 2168, 37, 62]; stop")

        return self.request(prompt, current_name)

    def fairness_rerank(self):
        current_name = 'fairness'
        prompt = "Considering a user, his/her basic infomation is: \n{"
        for fea in self.user_fea:
            prompt += "{}:{},".format(fea, str(self.data["user_info"][fea]))
        prompt += "}\nHis/Her history of browsing items and related features are: \n{"
        his_count = 1
        for _, item in self.data["history_items"].iterrows():
            prompt += "["
            for fea in self.his_item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += "]\n"
            his_count += 1
            if his_count > self.history_max:
                break
        prompt += "}\nHere's a list of the candidate items (with related features) he/she might see next: \n{"
        for _, item in self.data["candidate_items"].iterrows():
            prompt += "["
            for fea in self.item_fea:
                prompt += "{}:{},".format(fea, str(item[fea]))
            prompt += ("]\n")
        prompt += ("}\nNote that the 'score' feature is generated by an existing"
                   " recommendation model for your reference. ")

        if len(self.re_history[0]) > 0:
            prompt += "Your historical operations and reranking results represented by item_id are as follows: {\n"
            for i in range(len(self.re_history[0])):
                prompt += "Focus: {}. Reranking Result: [{}]\n".format(self.re_history[0][i],
                                                                       ", ".join(self.re_history[1][i]))
            prompt += "}\n"
        else:
            prompt += ("Your historical operations and reranking results represented by item_id are as follows: \n"
                       "{No historical operations yet.}\n")
        if self.dataset_name == 'ml-1m':
            prompt += (
                    "Now, you need to focus on the " + current_name + " aspect (For items with 'year' feature after "
                                                                      "1996 and items with 'year' feature before 1996, You should keep the average ranking of the two "
                                                                      "categories in the candidate items similar) ")
        elif self.dataset_name == 'kuairand':
            prompt += (
                    "Now, you need to focus on the " + current_name + " aspect (For items with different 'video_duration' feature "
                                                                      ", You should keep the average ranking of the two "
                                                                      "categories in the candidate items similar) ")
        prompt += ("and rerank the candidates based on the given "
                   "information, and then give suggestions about the next step of reranking from the following reranking "
                   "focus: \n{" + ', '.join(
            self.nodes) + "}\nSpecially, 'stop' means to stop reranking in the next step "
                          "and output the result of the current step as the final reranking output. And 'backward' means to "
                          "delete the latest reranking operation and result so that they are not taken into consideration by"
                          " subsequent operations.")

        prompt += (
                "\nThe order of reranking result should represent how likely the user is to watch it."
                "For suggestions about the next step, You should choose one of "
                "the functions representing the next reranking focus or 'stop' based on the final goal of the reranking:\n{" + self.focus + "}.\n"

                                                                                                                                            "For your response format, please only give me a list of item_id (containing all item_id values in the "
                                                                                                                                            "candidates list provided above and not containing any other item_id not mentioned) in order of recommendation priority and "
                                                                                                                                            "an operation name you suggest to do next from the list of reranking focus as your answer,"
                                                                                                                                            " and omit anything else such as your thinking and decision-making "
                                                                                                                                            "process. \nYour answer should strictly follow the format provided below containing a list of reranking result "
                                                                                                                                            "with all the item_id in the candidates and a word indicating the next operation. Punctuations marks not mentioned "
                                                                                                                                            "in the example should not be included in your answer. "
                                                                                                                                            "\nExample answer format for 20 candidates: \n[16, 1246, 536, 15, 748, 1135, 1636, "
                                                                                                                                            "1, 47, 8, 2478, 899, 151, 1032, 165, 363, 527, 2168, 37, 62]; stop")

        return self.request(prompt, current_name)
