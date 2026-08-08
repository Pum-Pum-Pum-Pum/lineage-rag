## r1-nfl-001

What can users configure when setting up a non-financial fee?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r1-nfl-002

Which tax and rounding applicability options are captured in Non-Financial Load maintenance?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r1-nfl-003

How is the transfer fee determined for transfer transactions?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r18-minor-program-001

When a Minor Program unit holder subscribes to a fund configured for the Minor Program, which bucket is consumed first?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r18-minor-program-002

What happens if a Minor Program subscription would breach the investment limit for the period?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r18-minor-program-003

What appears in BOR report B-12's Remark column for a Minor Program unit holder?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r21-fds-001

Which cheque statuses can be selected for Mark Status Change, and in which modules is this supported?

SME verdict: expected_case_incorrect
SME rationale: The Mark Status Change operation is enhanced in R21 to support additional cheque statuses in the HQ and FMG modules.

        Supported modules

        The enhancement applies to the Cheque Status Summary (UTDCHKST) screen in:

        HQ Module
        FMG Module

        The exception status changes are further restricted to ASNBHQ001 and FMGPNB modules.

        Cheque statuses that can be selected

        The enhancement enables Mark Status Change for the following current cheque statuses:

        Active
        Stop
        Cleared
        Rejected
        Expired
        Unclaimed
        Duplicate

        Previously, Mark Status Change was supported only when the current cheque status was Active.

        Existing behavior retained

        The existing functionality remains unchanged for Active cheques:

        Current Status: Active
        Allowed target statuses:
        Rejected
        Cleared
        Stop

        This behavior continues as before.

        Additional R21 enhancements

        The processing rules were relaxed so that, in HQ and FMG modules, Mark Status Change is also allowed when the current cheque status is:

        Cleared
        Expired
        Rejected
        Unclaimed
        Duplicate
        (plus the existing Active status)

        If another current status is selected, the system displays:

        "Cheque Status should be Active, Cleared, Expired, Rejected, Unclaimed and Duplicate when Operation is Mark Status".

        The document also specifies:

        Only valid status transition combinations are permitted; otherwise the system displays:

        "Status Change is not allowed for the selected Cheque Status and Mark Status".

        A Remarks field is introduced for exception status changes.
        All exception status changes can be viewed through UTDCHKST → View All Records.
        Certain transitions (for example, some transitions involving Duplicate) have additional business restrictions such as preventing cheque re-issue or disallowing changes when a re-issued cheque already exists.
        Source checked
        Metadata/index: R21 FDS Enhancement Phase 1 identified as the relevant candidate for "Change cheque status".
        Selected source FDD: FS_FCIS_14.7.0.0.0$ASNB_R21_FDS_Enhancement_Phase1_v1.4.docx
Required follow-up: Yes, it did not answer correctly

## r21-fds-002

What additional information appears on transaction draft prints for selected subscription transactions?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r21-fds-003

What controls apply when removing the HA/TD flag?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r24-reports-001

Before the Teller and Branch EOD report re-alignment, how many branch EOD reports existed, and how were they split?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r24-reports-002

What is the current name of teller report T-1?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r24-reports-003

What is the current name of branch report B-01?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## lineage-r2-001

Which document defines the original B-12 Account Registration/Closure Details report specification?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## lineage-r2-r18-002

What changed in the existing B-12 report's Remark column for Minor Program unit holders, and when was it introduced?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## lineage-r2-003

In what format is the T-2 Teller Transaction Summary report generated?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## lineage-r21-004

What is the current rule for changing a unit holder's status from Deceased to Normal?

SME verdict: conditionally accepted
SME rationale: It did answer correctly for the R21 latest update, originally 
    R2 (Death Claim) provides the original Deceased → Normal workflow related to death claims and Khairat reversal.
    R21 extends this capability by introducing a controlled Change UH Account Status process for correcting incorrect tagging, with authorization and role restrictions rather than a general status-edit capability. The metadata indicates this as a correction path, not a removal of the original R2 scenario.
Required follow-up: We should look for the latest change also look for the old working, only if the new overrided the old one, then we can say it is correct. Otherwise, we should state both old and new as they are both accurate.
## lineage-r21-005

Can the EPF Applicable field be amended for an existing minor unit holder, including through the REST API?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## lineage-r24-006

Which release introduced the Teller and Branch Reports Re-alignment, and how does it relate to the original Branch Online Reports specification?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-001

Does the Non-Financial Load enhancement define Minor Program investment-limit validation, or is that governed elsewhere?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-002

Can a cheque in Cleared status be selected for Mark Status Change, and where is this supported?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-003

Does the Teller and Branch report re-alignment introduce the B-12 Minor Program Remark-column behavior, or is it governed elsewhere?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-004

Is the original branch report produced in PDF, and what is the current name of teller report T-1?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-005

What enhancement added FE/LG and fundwise guardian information to transaction draft printing?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## confusion-release-006

What restrictions apply to the EPF Applicable field for minor unit holders?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No

## r18-minor-program-reinvestment-consumption-001

How are dividend reinvestment units consumed for Minor Program unit holders?

SME verdict: accepted
SME rationale: Answer is accurate and supported by the FDD
Required follow-up: No
