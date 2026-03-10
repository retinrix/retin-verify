"""
Synthetic Identity Generator for Algerian Documents
Generates fake but realistic Algerian identity data for synthetic document generation.
"""

import random
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import string


class AlgerianIdentityGenerator:
    """Generate synthetic Algerian identity data for document templates."""
    
    # Algerian first names (French/Latin + Arabic)
    FIRST_NAMES_MALE = [
        ('Mohamed', 'محمد'), ('Ahmed', 'أحمد'), ('Abdelkader', 'عبد القادر'), 
        ('Karim', 'كريم'), ('Sofiane', 'سفيان'), ('Amine', 'أمين'), 
        ('Yacine', 'ياسين'), ('Hichem', 'هشام'), ('Nadir', 'نذير'), 
        ('Rachid', 'رشيد'), ('Khaled', 'خالد'), ('Youcef', 'يوسف'), 
        ('Omar', 'عمر'), ('Brahim', 'إبراهيم'), ('Djamel', 'جمال'), 
        ('Fares', 'فارس'), ('Hamza', 'حمزة'), ('Islam', 'إسلام'), 
        ('Lotfi', 'لطفي'), ('Mehdi', 'مهدي'), ('Nabil', 'نبيل'),
        ('Salim', 'سليم'), ('Tarek', 'طارق'), ('Walid', 'وليد'), 
        ('Zinedine', 'زين الدين'), ('Adel', 'عادل'), ('Anis', 'أنيس'), 
        ('Faouzi', 'فوزي'), ('Hakim', 'حكيم'), ('Kamel', 'كمال'), 
        ('Mourad', 'مراد'), ('Nasreddine', 'نصر الدين'), ('Riad', 'رياض'), 
        ('Samir', 'سمير'), ('Toufik', 'توفيق')
    ]
    
    FIRST_NAMES_FEMALE = [
        ('Fatima', 'فاطمة'), ('Nadia', 'نادية'), ('Karima', 'كريمة'), 
        ('Samira', 'سميرة'), ('Leila', 'ليلى'), ('Amina', 'أمينة'), 
        ('Yasmine', 'ياسمين'), ('Asma', 'أسماء'), ('Djamila', 'جميلة'), 
        ('Fatiha', 'فتحية'), ('Hafida', 'حفيظة'), ('Kahina', 'الكاهنة'), 
        ('Lamia', 'لمياء'), ('Meriem', 'مريم'), ('Nawel', 'نوال'), 
        ('Rania', 'رانيا'), ('Sabrina', 'صابرين'), ('Wafa', 'وفاء'), 
        ('Zohra', 'زهرة'), ('Amel', 'أمل'), ('Dounia', 'دنيا'),
        ('Farida', 'فريدة'), ('Houda', 'هدى'), ('Imene', 'إيمان'), 
        ('Kaouthar', 'كوثر'), ('Lina', 'لينا'), ('Malika', 'مالكة'), 
        ('Nassima', 'نصيما'), ('Rachida', 'رشيدة'), ('Sarah', 'سارة'), 
        ('Souad', 'سعاد'), ('Yasmina', 'ياسمينة'), ('Zineb', 'زينب'), 
        ('Fadela', 'فضيلة'), ('Khadija', 'خديجة')
    ]
    
    # Algerian last names (French/Latin + Arabic)
    LAST_NAMES = [
        ('Benali', 'بن علي'), ('Said', 'سعيد'), ('Bouaziz', 'بوعزيز'), 
        ('Bensalem', 'بن سالم'), ('Ouali', 'والي'), ('Amrani', 'عمراني'), 
        ('Belaid', 'بلعيد'), ('Boudiaf', 'بوضياف'), ('Cherif', 'شريف'), 
        ('Djebbar', 'جبار'), ('Ferhat', 'فرحات'), ('Gacem', 'قاسم'), 
        ('Hamidi', 'حميدي'), ('Ibrahim', 'إبراهيم'), ('Kaci', 'قاسي'), 
        ('Lounes', 'وناس'), ('Madoui', 'مضوي'), ('Nait', 'نait'), 
        ('Oukid', 'وكيد'), ('Sahraoui', 'صحراوي'), ('Taleb', 'طالب'),
        ('Zerrouki', 'زروقي'), ('Abbas', 'عباس'), ('Ait', 'أيت'), 
        ('Belhadj', 'بلحاج'), ('Chikhi', 'شيخي'), ('Draoui', 'دراوي'), 
        ('Ghilani', 'غيلاني'), ('Haddad', 'حداد'), ('Idris', 'إدريس'), 
        ('Khalfallah', 'خلف الله'), ('Laouar', 'لعوار'), ('Mebarki', 'مباركي'), 
        ('Naceri', 'ناصري'), ('Selmani', 'سلماني')
    ]
    
    # Wilayas with Arabic names
    WILAYAS_AR = {
        'Alger': 'الجزائر', 'Oran': 'وهران', 'Constantine': 'قسنطينة', 
        'Annaba': 'عنابة', 'Blida': 'البليدة', 'Setif': 'سطيف', 
        'Tlemcen': 'تلمسان', 'Batna': 'باتنة', 'Djelfa': 'الجلفة', 
        'Skikda': 'سكيكدة', 'Biskra': 'بسكرة', 'Tiaret': 'تيارت',
        'Bejaia': 'بجاية', 'Tizi Ouzou': 'تيزي وزو', 'Mostaganem': 'مستغانم',
        'Boumerdes': 'بومرداس', 'Chlef': 'الشلف', 'Saida': 'سعيدة',
        'Mascara': 'معسكر', 'Ouargla': 'ورقلة', 'El Oued': 'الوادي',
        'Laghouat': 'الأغواط', 'Guelma': 'قالمة', 'Jijel': 'جيجل',
        'Medea': 'المدية', 'Tebessa': 'تبسة', 'Khenchela': 'خنشلة',
        'Souk Ahras': 'سوق أهراس', 'Mila': 'ميلة', 'Relizane': 'غليزان',
        'El Bayadh': 'البيض', 'Illizi': 'إيليزي', 'Bordj Bou Arreridj': 'برج بوعريريج',
        'Bouira': 'البويرة', 'Tamanrasset': 'تمنراست', 'Tindouf': 'تندوف',
        'Ain Temouchent': 'عين تموشنت', 'Ghardaia': 'غرداية', 'Naama': 'النعامة',
        'Ain Defla': 'عين الدفلى', 'Tipaza': 'تيبازة', 'M Sila': 'المسيلة',
        'El Tarf': 'الطارف', 'Tissemsilt': 'تسمسيلت', 'Khemis Miliana': 'خميس مليانة',
        'Oum El Bouaghi': 'أم البواقي', 'Sidi Bel Abbes': 'سيدي بلعباس', 
        'Adrar': 'أدرار'
    }
    
    # Algerian wilayas (provinces)
    WILAYAS = [
        'Alger', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Setif', 'Tlemcen',
        'Batna', 'Djelfa', 'Skikda', 'Biskra', 'Tiaret', 'Bejaia', 'Tizi Ouzou',
        'Mostaganem', 'Boumerdes', 'Chlef', 'Saida', 'Mascara', 'Ouargla',
        'El Oued', 'Laghouat', 'Guelma', 'Jijel', 'Medea', 'Tebessa', 'Khenchela',
        'Souk Ahras', 'Mila', 'Relizane', 'El Bayadh', 'Illizi', 'Bordj Bou Arreridj',
        'Bouira', 'Tamanrasset', 'Tindouf', 'Ain Temouchent', 'Ghardaia', 'Naama',
        'Ain Defla', 'Tipaza', 'M Sila', 'El Tarf', 'Tissemsilt', 'Khemis Miliana',
        'Oum El Bouaghi', 'Sidi Bel Abbes', 'M Sila', 'Oran', 'Adrar'
    ]
    
    # Blood groups
    BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
    
    def generate_date_of_birth(self, min_age: int = 18, max_age: int = 80) -> datetime:
        """Generate a random date of birth for a person within age range."""
        days_min = min_age * 365
        days_max = max_age * 365
        days_ago = random.randint(days_min, days_max)
        dob = datetime.now() - timedelta(days=days_ago)
        return dob
    
    def generate_document_dates(self, dob: datetime) -> tuple:
        """Generate issue and expiry dates based on date of birth."""
        # Issue date: sometime after 18th birthday
        min_issue = dob + timedelta(days=18*365 + random.randint(0, 365*20))
        max_issue = datetime.now() - timedelta(days=random.randint(0, 365*2))
        
        if min_issue > max_issue:
            issue_date = min_issue
        else:
            issue_date = min_issue + timedelta(
                seconds=random.randint(0, int((max_issue - min_issue).total_seconds()))
            )
        
        # Expiry date: 10 years for passport, varies for other docs
        expiry_date = issue_date + timedelta(days=365*10)
        
        return issue_date, expiry_date
    
    def calculate_mrz_check_digit(self, data: str) -> str:
        """Calculate MRZ check digit according to ICAO 9303."""
        weights = [7, 3, 1]
        total = 0
        
        for i, char in enumerate(data):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char) - ord('A') + 10
            elif char == '<':
                value = 0
            else:
                continue
            
            total += value * weights[i % 3]
        
        return str(total % 10)
    
    def generate_passport_number(self) -> str:
        """Generate a synthetic Algerian passport number."""
        # Format: DZ followed by 7 digits
        return f"DZ{random.randint(1000000, 9999999)}"
    
    def generate_national_id(self) -> str:
        """Generate a synthetic Algerian national ID number (NIR)."""
        # 9-digit national ID with checksum
        base = random.randint(10000000, 99999999)
        checksum = base % 97
        return f"{base}{checksum:02d}"
    
    def generate_personal_id(self) -> str:
        """Generate a synthetic personal ID number (18 digits like CNIE)."""
        # Format: 18 digits (as seen on CNIE: 109671329005850003)
        return ''.join([str(random.randint(0, 9)) for _ in range(18)])
    
    def generate_mrz(self, doc_type: str, identity: Dict) -> Dict[str, str]:
        """Generate MRZ lines for passport or CNIE."""
        if doc_type == 'passport':
            return self._generate_passport_mrz(identity)
        elif doc_type == 'cnie':
            return self._generate_cnie_mrz(identity)
        return {}
    
    def _generate_passport_mrz(self, identity: Dict) -> Dict[str, str]:
        """Generate passport MRZ according to ICAO 9303 TD3 format."""
        # MRZ always uses Latin/French names
        surname = identity['surname'].upper().replace(' ', '<')
        given_names = identity['given_names'].upper().replace(' ', '<')
        
        # Line 1: P<DZA{surname}<<{given_names}<<<<<<<<<<<<<<<
        name_part = f"{surname}<<{given_names}"
        line1 = f"P<DZA{name_part:<39}"
        line1 = line1[:44]
        
        # Line 2: {passport_num}DZA{dob}{sex}{expiry}{personal_num}{check_digit}
        pp_num = identity['passport_number'].replace('DZ', '')
        dob_str = identity['date_of_birth'].strftime('%y%m%d')
        expiry_str = identity['date_of_expiry'].strftime('%y%m%d')
        sex = identity['sex']
        
        # Calculate check digits
        pp_check = self.calculate_mrz_check_digit(pp_num)
        dob_check = self.calculate_mrz_check_digit(dob_str)
        expiry_check = self.calculate_mrz_check_digit(expiry_str)
        
        # Composite check digit
        composite = f"{pp_num}{pp_check}{dob_str}{dob_check}{expiry_str}{expiry_check}"
        composite_check = self.calculate_mrz_check_digit(composite)
        
        line2 = f"{pp_num}{pp_check}DZA{dob_str}{dob_check}{sex}{expiry_str}{expiry_check}<<<<<<<<<<{composite_check}"
        line2 = line2[:44]
        
        return {'line1': line1, 'line2': line2}
    
    def _generate_cnie_mrz(self, identity: Dict) -> Dict[str, str]:
        """Generate CNIE MRZ (simplified ID-1 format)."""
        id_num = identity['national_id'][:9]
        surname = identity['surname'].upper().replace(' ', '<')
        given_names = identity['given_names'].upper().replace(' ', '<')
        
        dob_str = identity['date_of_birth'].strftime('%y%m%d')
        expiry_str = identity['date_of_expiry'].strftime('%y%m%d')
        sex = identity['sex']
        
        line1 = f"I<DZA{id_num}<<<<<<<<<<<<<<<{surname}<<{given_names}"
        line1 = line1[:30]
        
        line2 = f"{dob_str}{sex}{expiry_str}DZA<<<<<<<<<<<0"
        line2 = line2[:30]
        
        return {'line1': line1, 'line2': line2}
    
    def generate_identity(self, doc_type: str = 'passport') -> Dict:
        """Generate a complete synthetic identity with bilingual names."""
        is_male = random.choice([True, False])
        
        # Get bilingual names
        surname_fr, surname_ar = random.choice(self.LAST_NAMES)
        given_fr, given_ar = random.choice(
            self.FIRST_NAMES_MALE if is_male else self.FIRST_NAMES_FEMALE
        )
        
        # Get place of birth with Arabic
        pob_fr = random.choice(list(self.WILAYAS_AR.keys()))
        pob_ar = self.WILAYAS_AR[pob_fr]
        
        # Arabic sex value
        sex_ar = 'ذكر' if is_male else 'أنثى'
        
        identity = {
            'surname': surname_fr,
            'surname_ar': surname_ar,
            'given_names': given_fr,
            'given_names_ar': given_ar,
            'sex': 'M' if is_male else 'F',
            'sex_ar': sex_ar,
            'nationality': 'DZA',
            'nationality_ar': 'جزائرية',
            'place_of_birth': pob_fr,
            'place_of_birth_ar': pob_ar,
            'blood_group': random.choice(self.BLOOD_GROUPS),
        }
        
        # Generate dates
        dob = self.generate_date_of_birth()
        issue_date, expiry_date = self.generate_document_dates(dob)
        
        identity['date_of_birth'] = dob
        identity['date_of_issue'] = issue_date
        identity['date_of_expiry'] = expiry_date
        
        # Document-specific fields
        if doc_type == 'passport':
            identity['document_number'] = self.generate_passport_number()
            identity['passport_number'] = identity['document_number']
        elif doc_type == 'cnie':
            identity['document_number'] = self.generate_national_id()
            identity['national_id'] = identity['document_number']
            identity['personal_id'] = self.generate_personal_id()
            place_fr = random.choice(list(self.WILAYAS_AR.keys()))
            identity['place_of_issue'] = place_fr
            identity['place_of_issue_ar'] = self.WILAYAS_AR[place_fr]
        
        # Generate MRZ
        identity['mrz'] = self.generate_mrz(doc_type, identity)
        
        return identity
    
    def generate_carte_grise_identity(self) -> Dict:
        """Generate synthetic vehicle registration data."""
        is_company = random.random() < 0.2  # 20% company-owned
        
        if is_company:
            owner_name = random.choice([
                'SARL TRANSPORT', 'ETS MOHAMED', 'SPA ALGERIE', 
                'SARL LOGISTICS', 'ETS SAID TRANSPORT'
            ])
            first_name = ''
        else:
            is_male = random.choice([True, False])
            owner_name = random.choice(self.LAST_NAMES)
            first_name = random.choice(
                self.FIRST_NAMES_MALE if is_male else self.FIRST_NAMES_FEMALE
            )
        
        # Generate VIN (17 characters)
        vin_chars = string.ascii_uppercase.replace('I', '').replace('O', '').replace('Q', '') + string.digits
        vin = ''.join(random.choices(vin_chars, k=17))
        
        # Registration number (Algerian format)
        reg_number = f"{random.randint(10000, 99999)} {random.choice(['A', 'B', 'C', 'D'])} {random.randint(1, 48)}"
        
        first_reg_date = datetime.now() - timedelta(days=random.randint(365, 365*20))
        
        return {
            'registration_number': reg_number,
            'first_registration_date': first_reg_date,
            'owner_surname': owner_name,
            'owner_given_names': first_name,
            'owner_address': f"{random.randint(1, 200)} Rue {random.choice(['Mohamed', 'Ahmed', 'Ali', 'Hassan'])}, {random.choice(self.WILAYAS)}",
            'vin': vin,
            'make': random.choice(['Renault', 'Peugeot', 'Hyundai', 'Toyota', 'Volkswagen', 'Dacia', 'Ford']),
            'type': random.choice(['Berline', 'SUV', 'Utilitaire', 'Camionnette', 'Break']),
            'commercial_designation': random.choice(['Clio', '208', 'i30', 'Corolla', 'Golf', 'Logan', 'Focus']),
            'national_vehicle_type': random.choice(['M1', 'N1', 'M2']),
            'max_laden_mass': random.choice(['1200', '1500', '2000', '3500']),
            'vehicle_mass': random.choice(['950', '1100', '1350', '1800']),
            'validity_period': '10 ans',
            'certificate_date': datetime.now() - timedelta(days=random.randint(0, 365*2)),
        }
    
    def generate_dataset(
        self, 
        doc_type: str, 
        num_samples: int,
        output_path: Optional[Path] = None
    ) -> List[Dict]:
        """Generate a dataset of synthetic identities."""
        dataset = []
        
        for i in range(num_samples):
            if doc_type == 'carte_grise':
                identity = self.generate_carte_grise_identity()
            else:
                identity = self.generate_identity(doc_type)
            
            identity['sample_id'] = i
            identity['document_type'] = doc_type
            dataset.append(identity)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert dates to strings for JSON serialization
            serializable_dataset = []
            for item in dataset:
                item_copy = item.copy()
                for key, value in item_copy.items():
                    if isinstance(value, datetime):
                        item_copy[key] = value.strftime('%d/%m/%Y')
                serializable_dataset.append(item_copy)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_dataset, f, indent=2, ensure_ascii=False)
        
        return dataset


if __name__ == '__main__':
    # Example usage
    generator = AlgerianIdentityGenerator(seed=42)
    
    # Generate sample identities
    print("=== Sample Passport Identity ===")
    passport_id = generator.generate_identity('passport')
    print(f"Name: {passport_id['surname']} {passport_id['given_names']}")
    print(f"DOB: {passport_id['date_of_birth'].strftime('%d/%m/%Y')}")
    print(f"Passport: {passport_id['passport_number']}")
    print(f"MRZ Line 1: {passport_id['mrz']['line1']}")
    print(f"MRZ Line 2: {passport_id['mrz']['line2']}")
    
    print("\n=== Sample CNIE Identity ===")
    cnie_id = generator.generate_identity('cnie')
    print(f"Name: {cnie_id['surname']} {cnie_id['given_names']}")
    print(f"National ID: {cnie_id['national_id']}")
    print(f"Place of Issue (Ar): {cnie_id.get('place_of_issue_ar', 'N/A')}")
    print(f"Sex (Ar): {cnie_id.get('sex_ar', 'N/A')}")
    print(f"Blood Group: {cnie_id['blood_group']}")
    
    print("\n=== Sample Carte Grise ===")
    cg_id = generator.generate_carte_grise_identity()
    print(f"Owner: {cg_id['owner_surname']} {cg_id['owner_given_names']}")
    print(f"Registration: {cg_id['registration_number']}")
    print(f"VIN: {cg_id['vin']}")
    print(f"Make: {cg_id['make']} {cg_id['commercial_designation']}")
    
    # Generate full dataset
    print("\n=== Generating Dataset ===")
    generator.generate_dataset('passport', 100, 'synthetic/output/passport_identities.json')
    generator.generate_dataset('cnie', 100, 'synthetic/output/cnie_identities.json')
    generator.generate_dataset('carte_grise', 100, 'synthetic/output/carte_grise_identities.json')
    print("Datasets saved to synthetic/output/")