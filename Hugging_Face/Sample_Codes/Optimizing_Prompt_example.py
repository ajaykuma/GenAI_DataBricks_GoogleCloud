from openai import OpenAI

# Initialize the client (make sure your OPENAI_API_KEY is set as an environment variable)
client = OpenAI()

# Ask user for product-specific inputs
brand_name = input("Enter the brand name: ")
product_name = input("Enter the product name: ")
product_category = input("Enter the product category (e.g., drink, oil, cream, bar): ")
product_description = input("Enter a short description of the product: ")
key_features = input("List the key product features (comma-separated): ")
revenue_target = input("Enter the target revenue increase (e.g., '24%'): ")
target_age_range = input("Enter the target age range (e.g., '22–55 years'): ")
target_interests = input("Enter target audience interests (comma-separated): ")
target_pain_points = input("Enter key audience pain points: ")
campaign_focus = input("Enter the campaign focus (e.g., 'Brand Awareness', 'Pre-orders', 'Seasonal Launch'): ")

# Construct the dynamic prompt
prompt = f"""
You are a senior marketing strategist with 25+ years of experience in the wellness, fitness, and healthy lifestyle industry.

Create a comprehensive product launch campaign for a new product.

**Context:**
{brand_name} aims to strengthen its position in the wellness and fitness market and achieve at least a {revenue_target} revenue increase compared to the previous year.
The campaign should focus on brand differentiation, audience engagement, and conversion.

**Product Details:**
- Product Name: {product_name}
- Product Category: {product_category}
- Description: {product_description}
- Key Features: {key_features}

**Target Audience:**
- Age Range: {target_age_range}
- Interests: {target_interests}
- Pain Points: {target_pain_points}

**Competitor Context:**
Analyze 3 leading competitors in the same product category from major platforms (e.g., Amazon, health-food retailers, or direct-to-consumer brands).
Highlight:
- Strengths
- Weaknesses
- Opportunities for {brand_name} to differentiate.

**Campaign Focus:** {campaign_focus}

**Deliverables:**
1. Campaign strategy summary (positioning, message, and insights)
2. Tagline
3. Instagram post (visual concept + caption)
4. Call-to-action (for pre-orders or early adoption, including any limited-time offer)
5. Differentiation strategy and growth recommendations to meet the revenue goal.

**Tone & Style:** Confident, aspirational, and aligned with modern wellness branding — a balance of science-backed credibility and lifestyle inspiration.
"""

# Send the prompt to GPT-4.1
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a world-class marketing strategist and copywriter."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
)

# Print the AI-generated campaign
print("\n--- Generated Product Launch Campaign ---\n")
print(response.choices[0].message.content)
