# The final features should define from interim data
# Overall 16 numeircal features and 11 categorical features

numerical_cols = [
    'age',
    'annual_company_revenue',
    'months_as_customer',
    'total_spend',
    'num_purchases',
    'last_purchase_days_ago',
    'monthly_active_days',
    'avg_session_duration_min',
    'features_used_count',
    'api_calls_last_30_days',
    'product_tours_completed',
    'support_tickets_total',
    'avg_ticket_resolution_hours',
    'sat_score',
    'campaigns_received',
    'last_marketing_touch_days_ago'
    ]

categorical_cols = [
    'gender',                    
    'country', 
    'company_size',
    'industry',
    'job_title'	,
    'plan_type',
    'has_churned_before',
    'discount_used',
    'onboarding_completed',
    'attended_webinar',
    'social_media_engaged'
    ]

target_col = 'will_repurchase'

remove_cols = ['customer_id']
