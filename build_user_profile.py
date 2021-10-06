import json
from collections import OrderedDict
from tqdm import tqdm
import sys
import os
from os.path import join
import pickle
from text_parser import TextParser
import subprocess
import csv

PROFILE_KEYS = ["pets", "family_members", "relationship_partners", "possessions_extra", 
"attributes_extra", "genders", "orientations", "attributes", "attributes_extra", "places_lived", 
"places_lived_extra", "places_grew_up", "places_grew_up_extra", "favorites", "actions_extra"]

parser = TextParser()

def load_attributes(chunk, user_dict):
    """
    Given an extracted chunk, load appropriate attribtues from it.
    """

    # Is this chunk a possession/belonging?
    if chunk["kind"] == "possession" and chunk["noun_phrase"]:
        # Extract noun from chunk
        noun_phrase = chunk["noun_phrase"]
        noun_phrase_text = " ".join([w for w, t in noun_phrase])
        norm_nouns = " ".join([
            parser.normalize(w, t) \
            for w, t in noun_phrase if t.startswith("N")
        ])

        noun = next(
            (w for w, t in noun_phrase if t.startswith("N")), None
        )
        if noun:
            # See if noun is a pet, family member or a relationship partner
            pet = parser.pet_animal(noun)
            family_member = parser.family_member(noun)
            relationship_partner = parser.relationship_partner(noun)

            if pet:
                user_dict["pets"].add(pet)
            elif family_member:
                user_dict["family_members"].add(family_member)
            elif relationship_partner:
                user_dict["relationship_partners"].add(
                    relationship_partner
                )
            else:
                user_dict["possessions_extra"].add(norm_nouns)

    # Is this chunk an action?
    elif chunk["kind"] == "action" and chunk["verb_phrase"]:
        verb_phrase = chunk["verb_phrase"]
        verb_phrase_text = " ".join([w for w, t in verb_phrase])

        # Extract verbs, adverbs, etc from chunk
        norm_adverbs = [
            parser.normalize(w, t) \
            for w, t in verb_phrase if t.startswith("RB")
        ]
        adverbs = [w.lower() for w, t in verb_phrase if t.startswith("RB")]

        norm_verbs = [
            parser.normalize(w, t) \
            for w, t in verb_phrase if t.startswith("V")
        ]
        verbs = [w.lower() for w, t in verb_phrase if t.startswith("V")]

        prepositions = [w for w, t in chunk["prepositions"]]

        noun_phrase = chunk["noun_phrase"]

        noun_phrase_text = " ".join(
            [w for w, t in noun_phrase if t not in ["DT"]]
        )
        norm_nouns = [
            parser.normalize(w, t) \
            for w, t in noun_phrase if t.startswith("N")
        ]
        proper_nouns = [w for w, t in noun_phrase if t == "NNP"]
        determiners = [
            parser.normalize(w, t) \
            for w, t in noun_phrase if t.startswith("DT")
        ]

        prep_noun_phrase = chunk["prep_noun_phrase"]
        prep_noun_phrase_text = " ".join([w for w, t in prep_noun_phrase])
        pnp_prepositions = [
            w.lower() for w, t in prep_noun_phrase if t in ["TO", "IN"]
        ]
        pnp_norm_nouns = [
            parser.normalize(w, t) \
            for w, t in prep_noun_phrase if t.startswith("N")
        ]
        pnp_determiners = [
            parser.normalize(w, t) \
            for w, t in prep_noun_phrase if t.startswith("DT")
        ]

        full_noun_phrase = (
                noun_phrase_text + " " + prep_noun_phrase_text
        ).strip()

        # TODO - Handle negative actions (such as I am not...),
        # but for now:
        if any(
                w in ["never", "no", "not", "nothing", "neither"] \
                for w in norm_adverbs + determiners
        ):
            return

        # I am/was ...
        if (len(norm_verbs) == 1 and "be" in norm_verbs and
                not prepositions and noun_phrase):
            # Ignore gerund nouns for now
            if (
                    "am" in verbs and
                    any(n.endswith("ing") for n in norm_nouns)
            ):
                user_dict["attributes_extra"].add(
                    full_noun_phrase
                )
                return

            attribute = []
            for noun in norm_nouns:
                gender = None
                orientation = None
                if "was" or "am" in verbs:
                    gender = parser.gender(noun)
                    orientation = parser.orientation(noun)
                if gender:
                    user_dict["genders"].add(gender)
                elif orientation:
                    user_dict["orientations"].add(
                        orientation
                    )
                # Include only "am" phrases
                elif "was" or "am" in verbs:
                    attribute.append(noun)

            if attribute and (
                    (
                            # Include only attributes that end
                            # in predefined list of endings...
                            any(
                                a.endswith(
                                    parser.include_attribute_endings
                                ) for a in attribute
                            ) and not (
                            # And exclude...
                            # ...certain lone attributes
                            (
                                    len(attribute) == 1 and
                                    attribute[0] in parser.skip_lone_attributes and
                                    not pnp_norm_nouns
                            )
                            or
                            # ...predefined skip attributes
                            any(a in attribute for a in parser.skip_attributes)
                            or
                            # ...attributes that end in predefined
                            # list of endings
                            any(
                                a.endswith(
                                    parser.exclude_attribute_endings
                                ) for a in attribute
                            )
                    )
                    ) or
                    (
                            # And include special attributes with different endings
                            any(a in attribute for a in parser.include_attributes)
                    )
            ):
                user_dict["attributes"].add(
                    full_noun_phrase
                )
            elif attribute:
                user_dict["attributes_extra"].add(
                    full_noun_phrase
                )

        # I live(d) in ...
        elif "live" in norm_verbs and prepositions and norm_nouns:
            if any(
                    p in ["in", "near", "by"] for p in prepositions
            ) and proper_nouns:
                user_dict["places_lived"].add(
                    " ".join(prepositions) + " " + noun_phrase_text,
                )
            else:
                user_dict["places_lived_extra"].add(
                        " ".join(prepositions) + " " + noun_phrase_text
                )

        # I grew up in ...
        elif "grow" in norm_verbs and "up" in prepositions and norm_nouns:
            if any(
                    p in ["in", "near", "by"] for p in prepositions
            ) and proper_nouns:
                user_dict["places_grew_up"].add(
                        " ".join(
                            [p for p in prepositions if p != "up"]
                        ) + " " + noun_phrase_text
                )
            else:
                user_dict["places_grew_up_extra"].add(
                        " ".join(
                            [p for p in prepositions if p != "up"]
                        ) + " " + noun_phrase_text,
                )

        elif (
                len(norm_verbs) == 1 and "prefer" in norm_verbs and
                norm_nouns and not determiners and not prepositions
        ):
            user_dict["favorites"].add(full_noun_phrase)

        elif (len(norm_verbs) == 1 and "have" in norm_verbs and
                norm_nouns
        ):
            for noun in norm_nouns:
                pet = parser.pet_animal(noun)
                family_member = parser.family_member(noun)
                relationship_partner = parser.relationship_partner(noun)

                if pet:
                    user_dict["pets"].add(pet)
                elif family_member:
                    user_dict["family_members"].add(family_member)
                elif relationship_partner:
                    user_dict["relationship_partners"].add(
                        relationship_partner
                    )
                else:
                    user_dict["possessions_extra"].add(noun)

        elif norm_nouns:
            actions_extra = " ".join(norm_verbs)
            user_dict["actions_extra"].add(actions_extra)


def build_user_profile(output_dir, file_name):
    """
    input: {"user_name": str, "description": str, "comments": list(str)}
    """
    with open('outputfile.json') as fin:
       all_users = json.load(fin)

    results = []
    output_file = join(output_dir, file_name)
    print("=======processing user history comments==========")
    num_lines = int(subprocess.check_output("/usr/bin/wc -l {}".format(output_file), shell=True).split()[0])
    for user_dict in all_users:
        user_profile = OrderedDict()
        user_profile["user_name"] = user_dict["user_name"]
        for key in PROFILE_KEYS:
            user_profile[key] = set()
        if user_dict["description"] == "":
            user_comments = []
        else:
            user_comments = [user_dict["description"]]

        for comment in user_dict["comments"]:
            user_comments.append(comment)

        all_comment = " ".join(user_comments)
        (chunks, sentiments) = parser.extract_chunks(all_comment)
        for chunk in chunks:
            load_attributes(chunk, user_profile)

        for key in PROFILE_KEYS:
            user_profile[key] = list(user_profile[key])

        results.append(user_profile)

    keys, values = [], []
    for key, value in results[0].items():
        keys.append(key)
    
    for u in results:
        v = []
        for key, value in u.items():
            v.append(value)
        values.append(v)
    print(keys)
    with open("user_information.csv", "w") as outfile:
        csvwriter = csv.writer(outfile, delimiter=';')
        csvwriter.writerow(keys)
        for row in values:
            csvwriter.writerow(row)
        print("Processing user history comments finished!\n")

if "--output-dir" in sys.argv:
    idx = sys.argv.index("--output-dir")
    try:
        output_dir = sys.argv[idx+1]
    except:
        raise ValueError("No output dir given!")
else:
    # default path
    output_dir = ""

if not os.path.exists('outputfile.json'):
    print("Run get_post_histories.py first.")

build_user_profile(output_dir, "outputfile.json")
