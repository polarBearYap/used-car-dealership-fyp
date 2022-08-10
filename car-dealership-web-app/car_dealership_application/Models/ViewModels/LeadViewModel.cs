using CarDealershipWebApp.Models.Interfaces;
using System;
using System.ComponentModel.DataAnnotations;

namespace CarDealershipWebApp.Models.ViewModels
{
    public class LeadViewModel : LeadBluePrint
    {
        [Display(Name = "Name")]
        [Required(AllowEmptyStrings = false)]
        public string Name
        {
            get
            {
                return lead.Name;
            }
            set
            {
                lead.Name = value;
            }
        }

        [Display(Name = "Email Address")]
        [Required(AllowEmptyStrings = false)]
        public string Email
        {
            get
            {
                return lead.Email;
            }
            set
            {
                lead.Email = value;
            }
        }

        [Display(Name = "Phone Number")]
        [Required(AllowEmptyStrings = false)]
        public string PhoneNo
        {
            get
            {
                return lead.PhoneNo;
            }
            set
            {
                lead.PhoneNo = value;
            }
        }

        [Display(Name = "Don't Email")]
        public int DontEmail
        {
            get
            {
                return Array.IndexOf(DontEmailList, lead.DontEmail);
            }
            set
            {
                lead.DontEmail = DontEmailList[value];
            }
        }

        [Display(Name = "Don't Call")]
        public int DontCall
        {
            get
            {
                return Array.IndexOf(DontCallList, lead.DontCall);
            }
            set
            {
                lead.DontCall = DontCallList[value];
            }
        }

        [Display(Name = "Occupation")]
        public int Occupation
        {
            get
            {
                return Array.IndexOf(OccupationList, lead.Occupation);
            }
            set
            {
                lead.Occupation = OccupationList[value];
            }
        }

        [Display(Name = "Received Free Copy of Car Report")]
        public int ReceivedFreeCopy
        {
            get
            {
                return Array.IndexOf(ReceivedFreeCopyList, lead.ReceivedFreeCopy);
            }
            set
            {
                lead.ReceivedFreeCopy = ReceivedFreeCopyList[value];
            }
        }

        [Display(Name = "Lead Status")]
        public int Status
        {
            get
            {
                return Array.IndexOf(StatusList, lead.Status);
            }
            set
            {
                lead.Status = StatusList[value];
            }
        }

        [Display(Name = "Average Page View per Visit")]
        [Range(0, 30)]
        public double AvgPageViewPerVisit
        {
            get
            {
                return lead.AvgPageViewPerVisit;
            }
            set
            {
                lead.AvgPageViewPerVisit = value;
            }
        }

        [Display(Name = "Created Time")]
        [DataType(DataType.DateTime)]
        [DisplayFormat(ApplyFormatInEditMode = false, DataFormatString = "{0:dd/MM/yyyy HH:mm}")]
        public DateTime CreatedTimestamp
        {
            get
            {
                return lead.CreatedTimestamp;
            }
            set
            {
                lead.CreatedTimestamp = value;
            }
        }

        [Display(Name = "Predicted Lead Score")]
        [Range(0, 1)]
        public double PredictedScore
        {
            get
            {
                return lead.PredictedScore;
            }
            set
            {
                lead.PredictedScore = value;
            }
        }

        [Display(Name = "Total Site Visited")]
        [Range(0, 50)]
        public double TotalSiteVisit
        {
            get
            {
                return lead.TotalSiteVisit;
            }
            set
            {
                lead.TotalSiteVisit = value;
            }
        }

        [Display(Name = "Total Time Spend on Site")]
        [Range(0, 2500)]
        public double TotalTimeSpendOnSite
        {
            get
            {
                return lead.TotalTimeSpendOnSite;
            }
            set
            {
                lead.TotalTimeSpendOnSite = value;
            }
        }

        public Lead lead;

        public LeadViewModel()
        {
            lead = new Lead();
        }

        public LeadViewModel(Lead lead)
        {
            this.lead = lead;
        }
    }
}
