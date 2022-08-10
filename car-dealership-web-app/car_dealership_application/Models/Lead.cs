using CarDealershipWebApp.Models.Interfaces;
using System;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.ComponentModel.DataAnnotations;

namespace CarDealershipWebApp.Models
{
    public class Lead : LeadBluePrint
    {
        public int ID { get; set; }

        [Display(Name = "Name")]
        [Column("Name")]
        public string Name { get; set; }

        [Display(Name = "Email Address")]
        [Column("Email")]
        public string Email { get; set; }

        [Display(Name = "Phone Number")]
        [Column("PhoneNo")]
        public string PhoneNo { get; set; }

        [Display(Name = "Don't Email")]
        [Column("DontEmail")]
        public string DontEmail
        {
            get
            {
                return _dont_email;
            }
            set
            {
                if (DontEmailList.Contains(value))
                    _dont_email = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _dont_email;

        [Display(Name = "Don't Call")]
        [Column("DontCall")]
        public string DontCall
        {
            get
            {
                return _dont_call;
            }
            set
            {
                if (DontCallList.Contains(value))
                    _dont_call = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _dont_call;

        [Display(Name = "Occupation")]
        [Column("Occupation")]
        public string Occupation
        {
            get
            {
                return _occupation;
            }
            set
            {
                if (OccupationList.Contains(value))
                    _occupation = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _occupation;

        [Display(Name = "Received Free Report")]
        [Column("ReceivedFreeCopy")]
        public string ReceivedFreeCopy
        {
            get
            {
                return _received_free_copy;
            }
            set
            {
                if (ReceivedFreeCopyList.Contains(value))
                    _received_free_copy = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _received_free_copy;

        [Display(Name = "Lead Status")]
        public string Status
        {
            get
            {
                return _status;
            }
            set
            {
                if (StatusList.Contains(value))
                    _status = value;
                else
                    throw new InvalidOperationException($"Invalid value. {value} does not exists.");
            }
        }
        private string _status;

        [Display(Name = "Average Page View per Visit")]
        public double AvgPageViewPerVisit { get; set; }

        [Display(Name = "Time Created")]
        [DisplayFormat(DataFormatString = "{0:dd/MM/yyyy HH:mm}")]
        public DateTime CreatedTimestamp { get; set; }

        [Display(Name = "Predicted Lead Score")]
        public double PredictedScore { get; set; }

        [Display(Name = "Total Site Visited")]
        public double TotalSiteVisit { get; set; }

        [Display(Name = "Total Time Spend on Site")]
        public double TotalTimeSpendOnSite { get; set; }
    }
}
