"""
CM Dataset to CCD Converter
Converts Cognitive Models to Cognitive Case Descriptions with transcript and symptom classification
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from openai import AsyncOpenAI
import os
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CMToCCDConverter:
    def __init__(self, api_key: str = None):
        """Initialize converter with OpenRouter API key"""
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_KEY")
        )
        # OpenRouter model format
        self.model = "openai/gpt-4o-mini"
        
        # Patient counter for each classification
        self.patient_counter = {
            'depression': 0,
            'delusion': 0,
            'psychosis': 0
        }
        
    async def load_data(self, cm_dataset_path: str) -> List:
        """Load CM dataset"""
        with open(cm_dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_examples(self) -> Dict[str, List[Dict]]:
        """Load CCD and transcript examples from provided documents"""
        examples = {
            'ccd': {
                'delusion': [],
                'depression': [],
                'psychosis': []
            },
            'transcript': {
                'delusion': [],
                'depression': [],
                'psychosis': []
            }
        }
        
        # CCD examples from documents
        # Delusion CCDs (documents 1-3)
        examples['ccd']['delusion'].extend([
            {
                "life_history": "The client has a complex family dynamic marked by recurrent conflicts and strained relationships. She has issues with trust and boundaries among family members and associates, including some who struggle with alcoholism and a relative who has taken advantage of her generosity. The client expresses frustration with those around her and discusses a history of significant aggressive behavior and violence.",
                "core_beliefs": ["Helpless belief", "Unlovable belief"],
                "core_belief_description": [
                    "Helpless: I am helpless. I am powerless, weak, vulnerable.",
                    "Unlovable: I am unlovable. I am bound to be alone."
                ],
                "intermediate_beliefs": [
                    "People always take advantage of me.",
                    "I need to be assertive to not be a victim.",
                    "I must protect myself from being hurt by others.",
                    "I can't rely on anyone."
                ],
                "intermediate_beliefs_during_delusion": [
                    "When I'm depressed, I believe that I will always be alone because nobody cares, I'm unlovable, and it's useless to try because people will disappoint me."
                ],
                "coping_strategies": [
                    "Avoiding social interactions to prevent feeling overwhelmed by rage.",
                    "Isolating at home, often staying in the bedroom."
                ],
                "cognitive_models": [
                    {
                        "situation": "Considering reaching out to an estranged individual who has hurt me.",
                        "automatic_thoughts": "He may come back and say sorry, but it won't be genuine. It will be the same cycle all over again.",
                        "emotion": "disappointed; hurt",
                        "behavior": "Debates internally whether to re-initiate contact, ultimately choosing not to out of self-protection."
                    }
                ]
            }
        ])
        
        # Depression CCDs (documents 4-6) - using one example
        examples['ccd']['depression'].extend([
            {
                "life_history": "The Client has been dealing with depressive symptoms that have affected their daily functioning, particularly in the evenings and in managing their apartment. They have had a recent improvement in mood and concentration, possibly due to therapy and participation in activities with their grandson, which provides them a sense of normalcy.",
                "core_beliefs": "Helpless belief",
                "core_belief_description": "I am powerless, weak, vulnerable",
                "intermediate_beliefs": "I should be able to do this on my own.",
                "intermediate_beliefs_during_depression": "If I don't get better, it means I am failing.",
                "coping_strategies": "Engaging in activities with family, reading therapeutic materials, and trying to push themselves to do things even when they feel self-critical.",
                "cognitive_models": [
                    {
                        "situation": "Sitting on the couch and feeling unproductive",
                        "automatic_thoughts": "I'm not going to get better.",
                        "emotion": "sad/down/lonely/unhappy",
                        "behavior": "Continues to sit on the couch; ruminates about my failures"
                    }
                ]
            }
        ])
        
        # Psychosis CCDs (documents 7-9) - using one example
        examples['ccd']['psychosis'].extend([
            {
                "life_history": "The client has been experiencing frustration, anxiety, and anger stemming from familial relationships. Conversations with his parents about his wedding list have brought insights into his own pattern of seeking approval and unconditional love from family members who do not reciprocate, a behavior he identifies as similar to his father's.",
                "core_beliefs": ["Helpless belief", "Unlovable belief"],
                "core_belief_description": [
                    "Helpless: I am needy. I am powerless, weak, vulnerable. I am trapped.",
                    "Unlovable: I am unlovable. I am undesirable, unwanted. I am bound to be abandoned. I am bound to be alone."
                ],
                "intermediate_beliefs": [
                    "One must continuously seek familial approval and support to feel safe and valued.",
                    "Investing in unhealthy familial relationships is necessary despite ongoing distress and frustration."
                ],
                "intermediate_beliefs_during_psychosis": [
                    "During periods of low mood, I adopt more polarized views on family loyalty and support, feeling an increased sense of obligation to pursue approval, which exacerbates feelings of helplessness and rejection."
                ],
                "coping_strategies": [
                    "Engaging in conversations with family members about emotional support and seeking to change family dynamics, though this is often met with resistance.",
                    "Has begun to set boundaries concerning emotional involvement with certain family members and is slowly reducing communication as a form of self-preservation."
                ],
                "cognitive_models": [
                    {
                        "situation": "Observing father's relentless pursuit of approval from his sister.",
                        "automatic_thoughts": "Why can't I stop seeking approval in situations that I recognize as unhealthy? Am I a sick person just like my father?",
                        "emotion": "anxious, worried, fearful, scared, tense",
                        "behavior": "Revisits stressful relationships and workplace, seeking safety within family circle. Engages in reassurance seeking."
                    }
                ]
            }
        ])
        
        # Transcript examples
        examples['transcript']['delusion'].extend([
            {
                "Delusion": [
                    {
                        "topic": "Struggles with trust in relationships",
                        "description": "I have difficulty trusting others due to past experiences of being taken advantage of and betrayed, particularly by family members."
                    },
                    {
                        "topic": "Fear of being hurt again",
                        "description": "I feel the need to protect myself from being hurt by others, which leads to self-isolation and avoiding social interactions."
                    }
                ]
            }
        ])
        
        examples['transcript']['depression'].extend([
            {
                "depression": [
                    {
                        "topic": "Feeling slightly better but unsure why",
                        "description": "I've been feeling a little better this week, but I'm not exactly sure why."
                    },
                    {
                        "topic": "Waking up more easily but struggling with motivation",
                        "description": "I've been able to wake up more easily in the mornings, but I still struggle with motivation."
                    }
                ]
            }
        ])
        
        examples['transcript']['psychosis'].extend([
            {
                "psychosis": [
                    {
                        "topic": "Parallels with father's approval-seeking behavior",
                        "description": "I observe my father's relentless pursuit of approval from family members and struggle with recognizing similar patterns in my own behavior."
                    },
                    {
                        "topic": "Fear of rejection and abandonment",
                        "description": "I fear losing my father's support and struggle with feelings of abandonment when familial relationships become strained."
                    }
                ]
            }
        ])
        
        return examples
    
    def group_cms_by_character(self, cm_dataset: List) -> Dict[str, List[Dict]]:
        """Group CMs by character name"""
        grouped = defaultdict(list)
        
        for entry in cm_dataset:
            # Use 'name' field as character identifier
            character_name = entry.get('name')
            if character_name:
                grouped[character_name].append(entry)
        
        return dict(grouped)
    
    async def generate_ccd(self, cms: List[Dict], examples: Dict) -> Dict:
        """Generate CCD from grouped CMs"""
        
        # Prepare example CCDs
        example_ccds = []
        for disorder, ccd_list in examples['ccd'].items():
            for ccd in ccd_list[:1]:  # Take first example from each disorder
                example_ccds.append(json.dumps(ccd, indent=2))
        
        cms_text = json.dumps(cms, indent=2, ensure_ascii=False)
        examples_text = "\n\n---Example CCD---\n\n".join(example_ccds)
        
        prompt = f"""You are an expert in cognitive behavioral therapy and psychological assessment. 

I will provide you with multiple Cognitive Models (CMs) for the same character/patient. Your task is to synthesize these CMs into a single Cognitive Case Description (CCD).

**Input Cognitive Models:**
```json
{cms_text}
```

**Example CCD Format:**
{examples_text}

**Instructions:**
1. Analyze all the provided CMs for common patterns, beliefs, and behaviors
2. Synthesize the information into a comprehensive CCD following the example format
3. Extract and consolidate:
   - life_history: Synthesize the patient's background and experiences
   - core_beliefs: Identify fundamental beliefs (e.g., "Helpless belief", "Unlovable belief")
   - core_belief_description: Detailed descriptions of core beliefs
   - intermediate_beliefs: Conditional beliefs and rules for living
   - intermediate_beliefs_during_delusion/depression/psychosis: Beliefs during symptomatic episodes
   - coping_strategies: How the patient manages distress
   - cognitive_models: Select 1-3 representative situations with automatic thoughts, emotions, and behaviors

4. Ensure the CCD is coherent, comprehensive, and clinically accurate
5. Return ONLY valid JSON matching the example format, no additional text

Generate the CCD now:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert clinical psychologist specializing in cognitive behavioral therapy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from markdown code blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        return json.loads(result)
    
    async def generate_transcript_topics(self, ccd: Dict, examples: Dict) -> Dict:
        """Generate transcript topics from CCD"""
        
        # Get example transcripts
        example_transcripts = []
        for disorder, transcript_list in examples['transcript'].items():
            for transcript in transcript_list[:1]:
                example_transcripts.append(json.dumps(transcript, indent=2))
        
        ccd_text = json.dumps(ccd, indent=2, ensure_ascii=False)
        examples_text = "\n\n---Example Transcript---\n\n".join(example_transcripts)
        
        prompt = f"""You are an expert in cognitive behavioral therapy and psychological assessment.

I will provide you with a Cognitive Case Description (CCD). Your task is to generate likely conversation topics and descriptions that would emerge in therapy sessions based on this CCD.

**Input CCD:**
```json
{ccd_text}
```

**Example Transcript Format:**
{examples_text}

**Instructions:**
1. Based on the CCD's situations, beliefs, and coping strategies, infer 6-10 likely therapy conversation topics
2. Each topic should include:
   - topic: A concise title (5-10 words)
   - description: A detailed first-person statement the patient would make (1-3 sentences)
3. Topics should reflect the patient's perspective and core struggles
4. Organize under the appropriate key: "depression", "Delusion", or "psychosis"
5. Return ONLY valid JSON matching the example format

Generate the transcript topics now:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert clinical psychologist specializing in cognitive behavioral therapy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract JSON from markdown code blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        return json.loads(result)
    
    async def classify_symptoms(self, ccd: Dict) -> str:
        """Classify the primary symptom category: delusion, depression, or psychosis"""
        
        ccd_text = json.dumps(ccd, indent=2, ensure_ascii=False)
        
        prompt = f"""You are an expert clinical psychologist. Analyze the following Cognitive Case Description (CCD) and classify the primary symptom category.

**Input CCD:**
```json
{ccd_text}
```

**Classification Categories:**
1. **depression**: Major depressive symptoms including persistent sadness, hopelessness, loss of interest, fatigue, worthlessness, suicidal ideation
2. **delusion**: Delusional beliefs, paranoia, persecution complex, suspiciousness, unusual thought content, grandiosity
3. **psychosis**: Hallucinations, severe delusions, disorganized thinking, conceptual disorganization, severe reality distortion

**Instructions:**
1. Analyze the core beliefs, intermediate beliefs, cognitive models, and symptom patterns
2. Determine which category BEST fits the primary symptomatology
3. Consider the severity and nature of cognitive distortions
4. Respond with ONLY ONE WORD: "depression", "delusion", or "psychosis"

Classification:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert clinical psychologist specializing in psychiatric diagnosis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate classification
        valid_categories = ["depression", "delusion", "psychosis"]
        if classification not in valid_categories:
            # Default to depression if invalid
            print(f"Warning: Invalid classification '{classification}', defaulting to 'depression'")
            classification = "depression"
        
        return classification
    
    async def process_character(self, character_id: str, cms: List[Dict], examples: Dict) -> Dict:
        """Process a single character: generate CCD, transcript, and classification"""
        
        print(f"\nProcessing character: {character_id}")
        print(f"  - Number of CMs: {len(cms)}")
        
        # Generate CCD
        print("  - Generating CCD...")
        ccd = await self.generate_ccd(cms, examples)
        
        # Classify symptoms
        print("  - Classifying symptoms...")
        classification = await self.classify_symptoms(ccd)
        
        # Generate transcript
        print("  - Generating transcript topics...")
        transcript = await self.generate_transcript_topics(ccd, examples)
        
        print(f"  - Classification: {classification}")
        
        return {
            'character_id': character_id,
            'classification': classification,
            'ccd': ccd,
            'transcript': transcript,
            'original_cms': cms
        }
    
    def initialize_patient_counter(self, output_dir: str):
        """Check existing patient files and initialize counter"""
        output_path = Path(output_dir)
        
        for classification in ['depression', 'delusion', 'psychosis']:
            ccd_dir = output_path / "CCD" / classification
            
            if ccd_dir.exists():
                # Find all patient files
                patient_files = list(ccd_dir.glob("patient*.json"))
                
                if patient_files:
                    # Extract patient numbers
                    patient_numbers = []
                    for file in patient_files:
                        name = file.stem  # e.g., "patient1"
                        if name.startswith("patient"):
                            try:
                                num = int(name.replace("patient", ""))
                                patient_numbers.append(num)
                            except ValueError:
                                continue
                    
                    if patient_numbers:
                        self.patient_counter[classification] = max(patient_numbers)
                        print(f"   Found {len(patient_numbers)} existing {classification} patients (max: patient{max(patient_numbers)})")
    
    async def convert_dataset(self, cm_dataset_path: str, output_dir: str = "output"):
        """Main conversion pipeline"""
        
        print("="*60)
        print("CM to CCD Conversion Pipeline")
        print("="*60)
        
        # Initialize patient counter from existing files
        print("\n1. Checking existing patient files...")
        self.initialize_patient_counter(output_dir)
        
        # Load data
        print("\n2. Loading CM dataset...")
        cm_dataset = await self.load_data(cm_dataset_path)
        print(f"   Loaded {len(cm_dataset)} CM entries")
        
        # Load examples
        print("\n3. Loading CCD and transcript examples...")
        examples = self.load_examples()
        print(f"   Loaded examples for {len(examples['ccd'])} disorders")
        
        # Group by character
        print("\n4. Grouping CMs by character...")
        grouped_cms = self.group_cms_by_character(cm_dataset)
        print(f"   Found {len(grouped_cms)} unique characters")
        
        # Process each character
        print("\n5. Processing characters...")
        results = []
        
        for character_id, cms in grouped_cms.items():
            try:
                result = await self.process_character(character_id, cms, examples)
                results.append(result)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  ERROR processing {character_id}: {e}")
                continue
        
        # Save results
        print(f"\n6. Saving results to {output_dir}/...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save by classification with patient numbering
        classification_counts = defaultdict(int)
        
        for result in results:
            classification = result['classification']
            
            # Increment patient counter for this classification
            self.patient_counter[classification] += 1
            patient_number = self.patient_counter[classification]
            patient_filename = f"patient{patient_number}"
            
            classification_counts[classification] += 1
            
            character_id = result['character_id']
            
            # Save CCD
            ccd_dir = output_path / "CCD" / classification
            ccd_dir.mkdir(parents=True, exist_ok=True)
            with open(ccd_dir / f"{patient_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(result['ccd'], f, indent=2, ensure_ascii=False)
            
            # Save transcript
            transcript_dir = output_path / "transcript" / classification
            transcript_dir.mkdir(parents=True, exist_ok=True)
            with open(transcript_dir / f"{patient_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(result['transcript'], f, indent=2, ensure_ascii=False)
            
            print(f"   Saved {character_id} as {classification}/{patient_filename}")
        
        # Save summary
        summary = {
            'total_characters': len(results),
            'classification_distribution': dict(classification_counts),
            'results': [
                {
                    'character_id': r['character_id'],
                    'classification': r['classification'],
                    'num_original_cms': len(r['original_cms'])
                }
                for r in results
            ]
        }
        
        with open(output_path / "conversion_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE")
        print("="*60)
        print(f"\nResults:")
        print(f"  - Total characters processed: {len(results)}")
        print(f"  - Classification distribution:")
        for classification, count in classification_counts.items():
            print(f"    - {classification}: {count}")
        print(f"\nOutput saved to: {output_path.absolute()}")
        
        return results


async def main():
    """Main execution function"""
    
    # Initialize converter
    # Make sure to set OPENAI_API_KEY environment variable
    converter = CMToCCDConverter()
    
    # Path to CM dataset
    cm_dataset_path = "/home/station_06/Desktop/EmoAgent/config/Patient Psi CM Dataset.json"
    
    # Output directory
    output_dir = "/home/station_06/Desktop/EmoAgent/config"
    
    # Run conversion
    results = await converter.convert_dataset(cm_dataset_path, output_dir)
    
    return results


if __name__ == "__main__":
    # Run the conversion
    results = asyncio.run(main())